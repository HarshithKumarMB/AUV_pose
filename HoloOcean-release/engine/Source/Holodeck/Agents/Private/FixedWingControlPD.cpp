#include "FixedWingControlPD.h"
#include "Holodeck.h"

UFixedWingControlPD::UFixedWingControlPD(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
	, PositionController(FW_POS_P, 0, FW_POS_D)
	, RotationController(FW_ROT_P, 0, FW_ROT_D) {}

void UFixedWingControlPD::Execute(
	void* const CommandArray,
	void* const InputCommand,
	float		DeltaSeconds) {
	if (FixedWing == nullptr) {
		FixedWing = static_cast<AFixedWing*>(FixedWingController->GetPawn());
		if (FixedWing == nullptr) {
			UE_LOG(
				LogHolodeck,
				Error,
				TEXT("UFixedWingControlPD couldn't get FixedWing reference"));
			return;
		}

		FixedWing->EnableDamping();
	}

	// Apply gravity & buoyancy
	FixedWing->ApplyBuoyancyDragForce();

	float* InputCommandFloat = static_cast<float*>(InputCommand);
	float* CommandArrayFloat = static_cast<float*>(CommandArray);

	// ALL calculations here are done in HoloOcean frame & units.

	// Get desired information
	FVector DesiredPosition =
		FVector(InputCommandFloat[0], InputCommandFloat[1], InputCommandFloat[2]);
	FVector DesiredOrientation =
		FVector(InputCommandFloat[3], InputCommandFloat[4], InputCommandFloat[5]);

	// Get current COM (frame we're moving), velocity, orientation, & ang.
	// velocity
	FVector Position = FixedWing->RootMesh->GetCenterOfMass();
	Position = ConvertLinearVector(Position, UEToClient);

	FVector LinearVelocity = FixedWing->RootMesh->GetPhysicsLinearVelocity();
	LinearVelocity = ConvertLinearVector(LinearVelocity, UEToClient);

	FVector Orientation = RotatorToRPY(FixedWing->GetActorRotation());

	FVector AngularVelocity = FixedWing->RootMesh->GetPhysicsAngularVelocityInDegrees();
	AngularVelocity = ConvertAngularVector(AngularVelocity, NoScale);

	// Compute accelerations to apply
	FVector LinAccel, AngAccel;
	for (int i = 0; i < 3; i++) {
		LinAccel[i] = PositionController.ComputePIDDirect(
			DesiredPosition[i], Position[i], LinearVelocity[i], DeltaSeconds);
		LinAccel[i] = FMath::Clamp(
			LinAccel[i], -FW_CONTROL_MAX_LIN_ACCEL, FW_CONTROL_MAX_LIN_ACCEL);

		AngAccel[i] = RotationController.ComputePIDDirect(
			DesiredOrientation[i],
			Orientation[i],
			AngularVelocity[i],
			DeltaSeconds,
			true,
			true);
		AngAccel[i] = FMath::Clamp(
			AngAccel[i], -FW_CONTROL_MAX_ANG_ACCEL, FW_CONTROL_MAX_ANG_ACCEL);
	}

	// Feedback linearize torque with buoyancy torque
	FRotator rotation = FixedWing->GetActorRotation();

	FVector e3 = rotation.UnrotateVector(FVector(0, 0, 1));
	e3 = ConvertLinearVector(e3, NoScale);

	FVector COB = ConvertLinearVector(
		FixedWing->CenterBuoyancy - FixedWing->CenterMass, UEToClient);
	FVector tau = FixedWing->Volume * FixedWing->WaterDensity * 9.8
		* FVector::CrossProduct(e3, COB);

	AngAccel += tau;

	// Move from body to global frame (have to convert to UE coordinates first,
	// and then back)
	FVector before = FVector(AngAccel);
	AngAccel = ConvertAngularVector(AngAccel, ClientToUE);
	AngAccel = rotation.RotateVector(AngAccel);
	AngAccel = ConvertAngularVector(AngAccel, UEToClient);

	// Fill in with the PD Control
	// Command array is then passed to vehicle as acceleration & angular velocity
	for (int i = 0; i < 3; i++) {
		CommandArrayFloat[i] = LinAccel[i];
		CommandArrayFloat[i + 3] = AngAccel[i];
	}
}