#include "HoveringAUVControlPD.h"
#include "Holodeck.h"

UHoveringAUVControlPD::UHoveringAUVControlPD(
	const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
	, PositionController_X(AUV_POS_X_P, AUV_POS_X_I, AUV_POS_X_D)
	, PositionController_Y(AUV_POS_Y_P, AUV_POS_Y_I, AUV_POS_Y_D)
	, PositionController_Z(AUV_POS_Z_P, AUV_POS_Z_I, AUV_POS_Z_D)
	, RotationController_R(AUV_ROT_R_P, AUV_ROT_R_I, AUV_ROT_R_D)
	, RotationController_P(AUV_ROT_P_P, AUV_ROT_P_I, AUV_ROT_P_D)
	, RotationController_Y(AUV_ROT_Y_P, AUV_ROT_Y_I, AUV_ROT_Y_D) {}

void UHoveringAUVControlPD::Execute(
	void* const CommandArray,
	void* const InputCommand,
	float		DeltaSeconds) {
	if (HoveringAUV == nullptr) {
		HoveringAUV = static_cast<AHoveringAUV*>(HoveringAUVController->GetPawn());
		if (HoveringAUV == nullptr) {
			UE_LOG(
				LogHolodeck,
				Error,
				TEXT("UHoveringAUVControlPD couldn't get HoveringAUV reference"));
			return;
		}

		HoveringAUV->EnableDamping();
	}

	// Apply gravity, buoyancy, drag
	HoveringAUV->ApplyBuoyancyDragForce();

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
	FVector Position = HoveringAUV->RootMesh->GetCenterOfMass();
	Position = ConvertLinearVector(Position, UEToClient);

	FVector LinearVelocity = HoveringAUV->RootMesh->GetPhysicsLinearVelocity();
	LinearVelocity = ConvertLinearVector(LinearVelocity, UEToClient);

	FVector Orientation = RotatorToRPY(HoveringAUV->GetActorRotation());

	FVector AngularVelocity =
		HoveringAUV->RootMesh->GetPhysicsAngularVelocityInDegrees();
	AngularVelocity = ConvertAngularVector(AngularVelocity, NoScale);

	// Compute accelerations to apply
	FVector LinAccel, AngAccel;

	LinAccel[0] = PositionController_X.ComputePIDDirect(
		DesiredPosition[0], Position[0], LinearVelocity[0], DeltaSeconds);
	LinAccel[0] = FMath::Clamp(
		LinAccel[0], -AUV_CONTROL_MAX_LIN_ACCEL, AUV_CONTROL_MAX_LIN_ACCEL);
	LinAccel[1] = PositionController_Y.ComputePIDDirect(
		DesiredPosition[1], Position[1], LinearVelocity[1], DeltaSeconds);
	LinAccel[1] = FMath::Clamp(
		LinAccel[1], -AUV_CONTROL_MAX_LIN_ACCEL, AUV_CONTROL_MAX_LIN_ACCEL);
	LinAccel[2] = PositionController_Z.ComputePIDDirect(
		DesiredPosition[2], Position[2], LinearVelocity[2], DeltaSeconds);
	LinAccel[2] = FMath::Clamp(
		LinAccel[2], -AUV_CONTROL_MAX_LIN_ACCEL, AUV_CONTROL_MAX_LIN_ACCEL);

	AngAccel[0] = RotationController_R.ComputePIDDirect(
		DesiredOrientation[0],
		Orientation[0],
		AngularVelocity[0],
		DeltaSeconds,
		true,
		true);
	AngAccel[0] = FMath::Clamp(
		AngAccel[0], -AUV_CONTROL_MAX_ANG_ACCEL, AUV_CONTROL_MAX_ANG_ACCEL);
	AngAccel[1] = RotationController_P.ComputePIDDirect(
		DesiredOrientation[1],
		Orientation[1],
		AngularVelocity[1],
		DeltaSeconds,
		true,
		true);
	AngAccel[1] = FMath::Clamp(
		AngAccel[1], -AUV_CONTROL_MAX_ANG_ACCEL, AUV_CONTROL_MAX_ANG_ACCEL);
	AngAccel[2] = RotationController_Y.ComputePIDDirect(
		DesiredOrientation[2],
		Orientation[2],
		AngularVelocity[2],
		DeltaSeconds,
		true,
		true);
	AngAccel[2] = FMath::Clamp(
		AngAccel[2], -AUV_CONTROL_MAX_ANG_ACCEL, AUV_CONTROL_MAX_ANG_ACCEL);

	// Feedback linearize torque with buoyancy torque
	FRotator rotation = HoveringAUV->GetActorRotation();

	FVector e3 = rotation.UnrotateVector(FVector(0, 0, 1));
	// e3 = ConvertLinearVector(e3, NoScale);

	FVector COB = HoveringAUV->CenterBuoyancy - HoveringAUV->CenterMass;
	// COB = ConvertLinearVector(COB, UEToClient);
	FVector tau = HoveringAUV->Volume * HoveringAUV->WaterDensity * 9.81
		* FVector::CrossProduct(e3, COB);
	// tau = rotation.RotateVector(tau);
	tau = ConvertAngularVector(tau, UEToClient);

	AngAccel += tau;

	// FVector before = FVector(AngAccel);
	// // Move from body to global frame (have to convert to UE coordinates first,
	// and then back) AngAccel = ConvertAngularVector(AngAccel, ClientToUE);
	// AngAccel = rotation.RotateVector(AngAccel);
	// AngAccel = ConvertAngularVector(AngAccel, UEToClient);
	// Fill in with the PID Control
	// Command array is then passed to vehicle as acceleration & angular velocity
	for (int i = 0; i < 3; i++) {
		CommandArrayFloat[i] = LinAccel[i];
		CommandArrayFloat[i + 3] = AngAccel[i];
	}
}