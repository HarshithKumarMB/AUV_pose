#pragma once

#include "Holodeck.h"
#include "BlueROV2.h"
#include "HolodeckControlScheme.h"
#include "HolodeckPawnController.h"
#include "SimplePID.h"
#include <math.h>

#include "BlueROV2ControlThrusters.generated.h"

/**
 * UBlueROV2ControlThrusters
 */
UCLASS()
class HOLODECK_API UBlueROV2ControlThrusters : public UHolodeckControlScheme {
public:
	GENERATED_BODY()

	UBlueROV2ControlThrusters(const FObjectInitializer& ObjectInitializer);

	void Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds)
		override;

	/** NOTE: These go counter-clockwise, starting in front right

  * 0: Vertical Front Starboard Thruster
  * 1: Vertical Front Port Thruster
  * 2: Vertical Back  Port Thruster
  * 3: Vertical Back  Starboard Thruster
  * 4: Angled   Front Starboard Thruster
  * 5: Angled   Front Port Thruster
  * 6: Angled   Back  Port Thruster
  * 7: Angled   Back  Starboard Thruster
  */
	unsigned int GetControlSchemeSizeInBytes() const override {
		return 8 * sizeof(float);
	}

	void SetController(AHolodeckPawnController* const Controller) {
		BlueROV2Controller = Controller;
	};

private:
	AHolodeckPawnController* BlueROV2Controller;
	ABlueROV2*				 BlueROV2;
};
