{
  description = "AUV pose estimation in HoloOcean";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      # CUDA, for fitting the bathymetry GP. nixpkgs' own `torch` is built
      # without it, so `torch.cuda.is_available()` is False however good the
      # card is; `torch-bin` is upstream's wheel and ships the CUDA runtime.
      #
      # That wheel is unfree -- `unfreeRedistributable` plus NVIDIA's `issl`,
      # the CUDA SDK licence -- so it needs `allowUnfree`, which is a licence
      # decision and not merely a build flag. Setting it here scopes the
      # decision to this flake rather than to the machine.
      #
      # Only `torch` is overridden, not the whole tree: `cudaSupport = true`
      # would rebuild torch and its dependents from source, which is hours of
      # compilation for the same result.
      #
      # CUDA is pinned forward because this nixpkgs defaults to 12.9.7 while the
      # wheel wants cuda-bindings >= 13.0.3, and nixpkgs refuses to evaluate it
      # otherwise. That refusal is a real check rather than noise, so it is
      # answered by supplying a new enough CUDA rather than by setting
      # `problems.handlers` to ignore it.
      #
      # `cuda-bindings` is a Python package in its own right and is what the
      # version check actually reads, so overriding `torch-bin`'s `cudaPackages`
      # alone changes nothing -- it has to be overridden in the set, from which
      # `torch-bin` then picks it up.
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (pyFinal: pyPrev: {
                cuda-bindings = pyPrev.cuda-bindings.override {
                  cudaPackages = final.cudaPackages_13;
                };
                # The wheel's metadata pins `setuptools<82` and nixpkgs carries
                # 83, which fails a runtime-dependency check rather than
                # anything that matters: torch uses setuptools only for
                # `torch.utils.cpp_extension`, which nothing here calls, and it
                # imports and runs fine against 83.
                torch =
                  (pyFinal.torch-bin.override {
                    cudaPackages = final.cudaPackages_13;
                  }).overrideAttrs
                    (_: {
                      dontCheckRuntimeDeps = true;
                    });
              })
            ];
          })
        ];
      };

      # NOT pkgs.python3 -- that is 3.14 in this nixpkgs, and torch/gpytorch
      # are only built for 3.13.
      python = pkgs.python313;

      # Upstream is access-gated (GitHub account linked to Epic Games) and PyPI is
      # stuck at 0.5.8, so 2.3.0 can only come from the vendored client. It is an
      # unmodified copy of tag v2.3.0 -- see vendor/holoocean/VENDOR.md, and
      # scripts/update-vendor.sh to re-verify that.
      #
      # Two spellings of the same path: a Nix path for the build, and a
      # repo-relative string for the shell hook. Keep them in sync.
      holooceanDir = "vendor/holoocean/client";
      holooceanSrc = ./vendor/holoocean/client;

      holoocean = python.pkgs.buildPythonPackage {
        pname = "holoocean";
        version = "2.3.0";
        format = "setuptools";
        src = holooceanSrc;

        # The checked-in pyproject.toml is 0 bytes, which confuses the PEP 517
        # backend selection. setup.py is the real metadata.
        postPatch = "rm -f pyproject.toml";

        propagatedBuildInputs = with python.pkgs; [
          numpy
          scipy
          matplotlib
          posix-ipc
        ];

        # client/tests is upstream's simulator suite; it needs the 5 GB worlds
        # and a running Unreal binary.
        doCheck = false;
        pythonImportsCheck = [ "holoocean" ];
      };

      auv-pose = python.pkgs.buildPythonPackage {
        pname = "auv-pose";
        version = "0.1.0";
        pyproject = true;
        src = ./.;
        build-system = [ python.pkgs.setuptools ];
        dependencies = with python.pkgs; [
          numpy
          pandas
          torch
          gpytorch
          scikit-learn
        ];
        nativeCheckInputs = with python.pkgs; [ pytestCheckHook ];
        pythonImportsCheck = [ "auv_pose.estimation" ];
      };

      pyEnv = python.withPackages (ps: with ps; [
        numpy
        scipy
        matplotlib
        pandas
        torch
        gpytorch
        scikit-learn
        posix-ipc
        tkinter # plt.show() needs a backend
        pytest
        ruff
      ]);

      # From HoloOcean-release/docker/runtime/Dockerfile -- upstream's own
      # statement of what the packaged Unreal binary links against.
      simLibs = with pkgs; [
        libGL
        libGLU
        vulkan-loader
        libx11
        libxcb
        libxrandr
        libxinerama
        libxcursor
        libxi
        libsm
        libxext
        libxrender
        libxkbcommon
        alsa-lib
        openal
        # SDL dlopens libudev.so.1 for joystick hotplug. Absent, SDL_UDEV_Init
        # fails and SDL's own teardown segfaults in SDL_UDEV_DelCallback before
        # the window is ever created. Not in upstream's Dockerfile because
        # Ubuntu ships it by default.
        systemd
        fontconfig
        freetype
        zlib
        stdenv.cc.cc.lib
      ];

      # The devShells read holoocean from the working tree rather than from
      # packages.holoocean, so edits to the vendored client take effect without a
      # rebuild -- useful when debugging sensor code. Use packages.holoocean when you
      # want the built, immutable version.
      # $PWD first so auv_pose resolves to the working tree -- the equivalent of an
      # editable install, without needing one.
      # The world binaries are 5.2 GB, so they live with the other datasets
      # rather than in the working tree. Set HOLODECKPATH before entering the
      # shell to override.
      shellEnv = ''
        export HOLODECKPATH="''${HOLODECKPATH:-$HOME/data/holoocean}"
        export PYTHONPATH="$PWD:$PWD/${holooceanDir}/src:$PYTHONPATH"
      '';
      # The packaged Unreal binary is a foreign ELF that expects an FHS layout, so it
      # cannot run against the Nix store directly.
      simFhs = pkgs.buildFHSEnv {
        name = "auv-pose-sim";
        targetPkgs = _: [ pyEnv ] ++ simLibs;
        profile = shellEnv;
        runScript = "bash";
      };
    in
    {
      packages.${system} = {
        inherit holoocean auv-pose;
        default = pyEnv;
        sim = simFhs;
      };

      # Mirrors the dev shell so the two cannot disagree: experiments/ for the
      # tests that import it, vendor/ on PYTHONPATH because navigate.py imports
      # holoocean at module scope, and map*.csv because tests/test_soundings.py
      # asserts against the committed survey data.
      #
      # Importing holoocean does not start the simulator -- only holoocean.make
      # does -- and testpaths keeps collection to tests/, so nothing here can
      # launch a world.
      checks.${system}.pytest = pkgs.runCommand "auv-pose-pytest" { } ''
        cp -r ${./.}/{auv_pose,experiments,tests,vendor,pyproject.toml,map.csv,map1.csv} .
        chmod -R +w .
        export PYTHONPATH="$PWD:$PWD/${holooceanDir}/src"
        export HOME=$TMPDIR
        export MPLBACKEND=Agg
        ${pyEnv}/bin/pytest
        touch $out
      '';

      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [ pyEnv ];
          shellHook = shellEnv;
        };

        # Interactive use only. `nix develop .#sim --command ...` silently drops the
        # command, because buildFHSEnv's shellHook execs into the FHS bash. To run
        # something non-interactively, use the package instead:
        #     nix run .#sim -- -c "python experiments/navigate.py"
        sim = simFhs.env;
      };
    };
}
