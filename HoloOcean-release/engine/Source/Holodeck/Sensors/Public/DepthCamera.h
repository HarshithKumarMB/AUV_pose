#pragma once

#include "CameraSensor.h"
#include "Containers/Queue.h"
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Holodeck.h"

#include "DepthCamera.generated.h"
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class HOLODECK_API UDepthCamera : public UCameraSensor {
	GENERATED_BODY()

public:
	UDepthCamera();

	/**
	 * InitializeSensor
	 * Sets up the class
	 */
	virtual void InitializeSensor() override;

	/**
	 * Allows parameters to be set dynamically
	 */
	virtual void ParseSensorParms(FString ParmsJson) override;

protected:
	virtual void TickSensorComponent(
		float						 DeltaTime,
		ELevelTick					 TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

private:
	int		TickCounter = 0;
	AActor* Parent;
};