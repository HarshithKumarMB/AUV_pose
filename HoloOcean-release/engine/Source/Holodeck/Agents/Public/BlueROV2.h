// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#pragma once

#include "Containers/Array.h"
#include "GameFramework/Pawn.h"
#include "HolodeckBuoyantAgent.h"

#include "BlueROV2.generated.h"

const float BR_MAX_LIN_ACCEL = 10;
const float BR_MAX_ANG_ACCEL = 2;
const float BR_MAX_THRUST = BR_MAX_LIN_ACCEL * 11.5 / 4;

UCLASS()
/**
 * BlueROV2
 * Inherits from the HolodeckAgent class.
 * On any tick this object will apply the given forces.
 * Desired values must be set by a controller.
 */
class HOLODECK_API ABlueROV2 : public AHolodeckBuoyantAgent {
	GENERATED_BODY()

public:
	/**
	 * Default Constructor.
	 */
	ABlueROV2();

	void InitializeAgent() override;

	/**
	 * Tick
	 * Called each frame.
	 * @param DeltaSeconds the time since the last tick.
	 */
	void Tick(float DeltaSeconds) override;

	unsigned int GetRawActionSizeInBytes() const override { return 8 * sizeof(float); };
	void* GetRawActionBuffer() const override { return (void*)CommandArray; };

	// Allows agent to fall up to ~8 meters
	float GetAccelerationLimit() override { return 400; }

	// Location of all thrusters relative to mesh origin in UE units (cm, LHS)
	TArray<FVector> thrusterLocations{
		// clang-format off
		FVector(12.00, 21.81, 7.09),   
		FVector(12.00, -21.81, 7.09),
		FVector(-12.00, -21.81, 7.09), 
		FVector(-12.00, 21.81, 7.09),
		FVector(15.62, 9.88, -1.00),   
		FVector(15.62, -9.88, -1.00),
		FVector(-15.62, -9.88, -1.00),
		FVector(-15.62, 9.88, -1.00)
	}; // clang-format on

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = BuoyancySettings)
	bool Perfect = true;

	void ApplyThrusters(float* const ThrusterArray);

	void EnableDamping();

private:
	// Accelerations
	float CommandArray[8];
};
