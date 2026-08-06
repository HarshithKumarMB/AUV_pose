#pragma once

#include "CameraSensor.h"
#include "Holodeck.h"
#include "ShaderBasedSensor.h"

#include "SemanticSegmentationCamera.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class HOLODECK_API USemanticSegmentationCamera : public UShaderBasedSensor {
	GENERATED_BODY()

public:
	USemanticSegmentationCamera();

	/**
	 * InitializeSensor
	 * Sets up the class
	 **/
	virtual void InitializeSensor() override;

	/**
	 * Allows parameters to be set dynamically
	 **/
	virtual void ParseSensorParms(FString ParmsJson) override;

protected:
	virtual void TickSensorComponent(
		float						 DeltaTime,
		ELevelTick					 TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

private:
	int TickCounter = 0;
};