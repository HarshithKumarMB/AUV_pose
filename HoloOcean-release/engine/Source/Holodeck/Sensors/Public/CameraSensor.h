
#pragma once
#include "HolodeckCamera.h"

class ASceneCapture2D;
class UMaterial;

#include "Containers/Queue.h"
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Holodeck.h"
#include "HolodeckSensor.h"

#include "CameraSensor.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class HOLODECK_API UCameraSensor : public UHolodeckCamera {
	GENERATED_BODY()

public:
	UCameraSensor();

	/**
	 * InitializeSensor
	 * Sets up the class
	 */
	virtual void InitializeSensor() override;

	/**
	 * Allows parameters to be set dynamically
	 */
	virtual void ParseSensorParms(FString ParmsJson) override;

	UPROPERTY(EditAnywhere)
	int TicksPerCapture = 1;

	/// ================================================================================
	/// Section was borrowed from CARLA Simulator:
	/// // Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat
	/// Autonoma
	// de Barcelona (UAB).
	//
	// This work is licensed under the terms of the MIT license.
	// For a copy, see <https://opensource.org/licenses/MIT>.

	/// ================================================================================
	UFUNCTION(BlueprintCallable)
	void EnablePostProcessingEffects(bool Enable = true) {
		bEnablePostProcessingEffects = Enable;
	}

	UFUNCTION(BlueprintCallable)
	bool ArePostProcessingEffectsEnabled() const {
		return bEnablePostProcessingEffects;
	}

	UFUNCTION(BlueprintCallable)
	void Enable16BitFormat(bool Enable = false) { bEnable16BitFormat = Enable; }

	UFUNCTION(BlueprintCallable)
	bool Is16BitFormatEnabled() const { return bEnable16BitFormat; }

	UFUNCTION(BlueprintCallable)
	void SetFOVAngle(float FOVAngle);

	UFUNCTION(BlueprintCallable)
	float GetFOVAngle() const;

	float FovAngle;

	UFUNCTION(BlueprintCallable)
	void SetViewDistance(float viewDistance);

	UFUNCTION(BlueprintCallable)
	float GetViewDistance() const;

	float ViewDistance;

	UFUNCTION(BlueprintCallable)
	void SetTargetGamma(float InTargetGamma) { TargetGamma = InTargetGamma; }

	UFUNCTION(BlueprintCallable)
	float GetTargetGamma() const { return TargetGamma; }

	UFUNCTION(BlueprintCallable)
	void SetExposureMethod(EAutoExposureMethod Method);

	UFUNCTION(BlueprintCallable)
	EAutoExposureMethod GetExposureMethod() const;

	EAutoExposureMethod AutoExposureMethod;

	UFUNCTION(BlueprintCallable)
	void SetExposureCompensation(float Compensation);

	UFUNCTION(BlueprintCallable)
	float GetExposureCompensation() const;

	float ExposureCompensation;

	UFUNCTION(BlueprintCallable)
	void SetShutterSpeed(float Speed);

	UFUNCTION(BlueprintCallable)
	float GetShutterSpeed() const;

	float ShutterSpeed;

	UFUNCTION(BlueprintCallable)
	void SetISO(float ISO);

	UFUNCTION(BlueprintCallable)
	float GetISO() const;

	float Iso;

	UFUNCTION(BlueprintCallable)
	void SetAperture(float Aperture);

	UFUNCTION(BlueprintCallable)
	float GetAperture() const;

	float Aperture;

	UFUNCTION(BlueprintCallable)
	void SetFocalDistance(float Distance);

	UFUNCTION(BlueprintCallable)
	float GetFocalDistance() const;

	float FocalDistance;

	UFUNCTION(BlueprintCallable)
	void SetDepthBlurAmount(float Amount);

	UFUNCTION(BlueprintCallable)
	float GetDepthBlurAmount() const;

	float DepthBlurAmount;

	UFUNCTION(BlueprintCallable)
	void SetDepthBlurRadius(float Radius);

	UFUNCTION(BlueprintCallable)
	float GetDepthBlurRadius() const;

	float DepthBlurRadius;

	UFUNCTION(BlueprintCallable)
	void SetBladeCount(int Count);

	UFUNCTION(BlueprintCallable)
	int GetBladeCount() const;

	int BladeCount;

	UFUNCTION(BlueprintCallable)
	void SetDepthOfFieldMinFstop(float MinFstop);

	UFUNCTION(BlueprintCallable)
	float GetDepthOfFieldMinFstop() const;

	float DepthOfFieldMinFstop;

	UFUNCTION(BlueprintCallable)
	void SetFilmSlope(float Slope);

	UFUNCTION(BlueprintCallable)
	float GetFilmSlope() const;
	float FilmSlope;
	UFUNCTION(BlueprintCallable)
	void SetFilmToe(float Toe);

	UFUNCTION(BlueprintCallable)
	float GetFilmToe() const;
	float FilmToe;
	UFUNCTION(BlueprintCallable)
	void SetFilmShoulder(float Shoulder);

	UFUNCTION(BlueprintCallable)
	float GetFilmShoulder() const;
	float FilmShoulder;
	UFUNCTION(BlueprintCallable)
	void SetFilmBlackClip(float BlackClip);

	UFUNCTION(BlueprintCallable)
	float GetFilmBlackClip() const;
	float FilmBlackClip;
	UFUNCTION(BlueprintCallable)
	void SetFilmWhiteClip(float WhiteClip);

	UFUNCTION(BlueprintCallable)
	float GetFilmWhiteClip() const;
	float FilmWhiteClip;
	UFUNCTION(BlueprintCallable)
	void SetExposureMinBrightness(float Brightness);

	UFUNCTION(BlueprintCallable)
	float GetExposureMinBrightness() const;
	float ExposureMinBrightness;
	UFUNCTION(BlueprintCallable)
	void SetExposureMaxBrightness(float Brightness);

	UFUNCTION(BlueprintCallable)
	float GetExposureMaxBrightness() const;
	float ExposureMaxBrightness;
	UFUNCTION(BlueprintCallable)
	void SetExposureSpeedDown(float Speed);

	UFUNCTION(BlueprintCallable)
	float GetExposureSpeedDown() const;
	float ExposureSpeedDown;
	UFUNCTION(BlueprintCallable)
	void SetExposureSpeedUp(float Speed);

	UFUNCTION(BlueprintCallable)
	float GetExposureSpeedUp() const;
	float ExposureSpeedUp;

	UFUNCTION(BlueprintCallable)
	void SetMotionBlurIntensity(float Intensity);

	UFUNCTION(BlueprintCallable)
	float GetMotionBlurIntensity() const;
	float MotionBlurIntensity;
	UFUNCTION(BlueprintCallable)
	void SetMotionBlurMaxDistortion(float MaxDistortion);

	UFUNCTION(BlueprintCallable)
	float GetMotionBlurMaxDistortion() const;
	float MotionBlurMaxDistortion;
	UFUNCTION(BlueprintCallable)
	void SetMotionBlurMinObjectScreenSize(float ScreenSize);

	UFUNCTION(BlueprintCallable)
	float GetMotionBlurMinObjectScreenSize() const;
	float MotionBlurMinObjectScreenSize;
	UFUNCTION(BlueprintCallable)
	void SetLensFlareIntensity(float Intensity);

	UFUNCTION(BlueprintCallable)
	float GetLensFlareIntensity() const;
	float LensFlareIntensity;
	UFUNCTION(BlueprintCallable)
	void SetBloomIntensity(float Intensity);

	UFUNCTION(BlueprintCallable)
	float GetBloomIntensity() const;
	float BloomIntensity;
	UFUNCTION(BlueprintCallable)
	void SetWhiteTemp(float Temp);

	UFUNCTION(BlueprintCallable)
	float GetWhiteTemp() const;
	float WhiteTemp;
	UFUNCTION(BlueprintCallable)
	void SetWhiteTint(float Tint);

	UFUNCTION(BlueprintCallable)
	float GetWhiteTint() const;
	float WhiteTint;
	UFUNCTION(BlueprintCallable)
	void SetChromAberrIntensity(float Intensity);

	UFUNCTION(BlueprintCallable)
	float GetChromAberrIntensity() const;
	float ChromAberrIntensity;
	UFUNCTION(BlueprintCallable)
	void SetChromAberrOffset(float chromAberrOffset);

	UFUNCTION(BlueprintCallable)
	float GetChromAberrOffset() const;
	float ChromAberrOffset;
	UFUNCTION(BlueprintCallable)
	USceneCaptureComponent2D* GetCaptureComponent2D() { return SceneCapture; }

	UFUNCTION(BlueprintCallable)
	UTextureRenderTarget2D* GetCaptureRenderTarget() { return TargetTexture; }

	/// Immediate enqueues render commands of the scene at the current time.
	void EnqueueRenderSceneImmediate();

protected:
	// Checkout HolodeckSensor.h for the documentation on these overridden
	// functions.
	virtual void TickSensorComponent(
		float						 DeltaTime,
		ELevelTick					 TickType,
		FActorComponentTickFunction* ThisTickFunction);

	virtual int GetNumItems() { return CaptureWidth * CaptureHeight; };
	virtual int GetItemSize() { return sizeof(float); };

	UPROPERTY(EditAnywhere)
	float TargetGamma = 2.4f;

	/// Image width in pixels.
	UPROPERTY(EditAnywhere)
	uint32 ImageWidth = 800u;

	/// Image height in pixels.
	UPROPERTY(EditAnywhere)
	uint32 ImageHeight = 600u;

	/// Whether to render the post-processing effects present in the scene.
	UPROPERTY(EditAnywhere)
	bool bEnablePostProcessingEffects = true;

	/// Whether to change render target format to PF_A16B16G16R16, offering 16bit
	/// / channel
	UPROPERTY(EditAnywhere)
	bool bEnable16BitFormat = false;

private:
	int		TickCounter = 0;
	AActor* Parent;
};
