// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "BlueROV2.h"
#include "Holodeck.h"

// Sets default values
ABlueROV2::ABlueROV2() {
	PrimaryActorTick.bCanEverTick = true;

	// Set the defualt controller
	AIControllerClass = LoadClass<AController>(
		NULL, TEXT("/Script/Holodeck.BlueROV2Controller"), NULL, LOAD_None, NULL);
	AutoPossessAI = EAutoPossessAI::PlacedInWorld;

	this->Volume = .03554577;
	this->MassInKG = 11.5;
	this->CenterMass = FVector(
		0,
		0,
		3); // vector from mesh origin to center of mass, expressed in mesh frame
	this->CenterBuoyancy = FVector(0, 0, 5); // vector from mesh origin to center of
											 // buoyancy, expressed in mesh frame
	this->CoefficientOfDrag = 0.8;
	this->AreaOfDrag = .45;
}

// Sets all values that we need
void ABlueROV2::InitializeAgent() {
	RootMesh = Cast<UStaticMeshComponent>(RootComponent);

	if (Perfect) {
		this->CenterMass = (thrusterLocations[0] + thrusterLocations[2]) / 2;
		this->CenterMass.Z = thrusterLocations[7].Z;

		this->CenterBuoyancy = CenterMass;
		this->CenterBuoyancy.Z += 5;

		this->Volume = MassInKG / WaterDensity;
	}

	// Convert thruster locations to be relative to the center of mass (in mesh
	// frame)
	for (int i = 0; i < 8; i++) {
		thrusterLocations[i] -= CenterMass;
	}

	Super::InitializeAgent();
}

// Called every frame
void ABlueROV2::Tick(float DeltaSeconds) {
	Super::Tick(DeltaSeconds);

	// Convert linear acceleration to force
	FVector linAccel = FVector(CommandArray[0], CommandArray[1], CommandArray[2]);
	linAccel =
		ClampVector(linAccel, -FVector(BR_MAX_LIN_ACCEL), FVector(BR_MAX_LIN_ACCEL));
	linAccel = ConvertLinearVector(linAccel, ClientToUE);

	// Convert angular acceleration to torque
	FVector angAccel = FVector(CommandArray[3], CommandArray[4], CommandArray[5]);
	angAccel =
		ClampVector(angAccel, -FVector(BR_MAX_ANG_ACCEL), FVector(BR_MAX_ANG_ACCEL));
	angAccel = ConvertAngularVector(angAccel, NoScale);

	RootMesh->GetBodyInstance()->AddForce(linAccel, true, true);
	RootMesh->GetBodyInstance()->AddTorqueInRadians(angAccel, true, true);
}

// For empty dynamics, damping is disabled
// Enable it when using thrusters/controller
void ABlueROV2::EnableDamping() {
	RootMesh->SetLinearDamping(1.0);
	RootMesh->SetAngularDamping(0.75);
}

void ABlueROV2::ApplyThrusters(float* const ThrusterArray) {
	/// Iterate through vertical thrusters
	for (int i = 0; i < 8; i++) {
		float force = FMath::Clamp(ThrusterArray[i], -BR_MAX_THRUST, BR_MAX_THRUST);

		// Calculate force
		FVector LocalForce = FVector(0, 0, 0); // Newtons
		if (i < 4)
			LocalForce = FVector(0, 0, force);
		else if (i % 2 == 0)
			LocalForce = FVector(
				force / UKismetMathLibrary::Sqrt(2),
				force / UKismetMathLibrary::Sqrt(2),
				0);
		else
			LocalForce = FVector(
				force / UKismetMathLibrary::Sqrt(2),
				-force / UKismetMathLibrary::Sqrt(2),
				0);

		LocalForce = ConvertLinearVector(
			LocalForce, ClientToUE); // convert Newtons to cN, reverses y axis

		// Apply force in global frame at thruster location
		FRotator bodyToWorld = this->GetActorRotation();
		FVector	 WorldForce = bodyToWorld.RotateVector(
			 LocalForce); // rotate force from body frame to global frame
		FVector ThrusterWorld = RootMesh->GetCenterOfMass()
			+ bodyToWorld.RotateVector(
				thrusterLocations[i]); // get thruster location in global frame
		RootMesh->AddForceAtLocation(WorldForce, ThrusterWorld);
	}
}