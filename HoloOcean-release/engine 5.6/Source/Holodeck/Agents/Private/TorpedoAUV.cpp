// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "TorpedoAUV.h"
#include "Holodeck.h"

// Sets default values
ATorpedoAUV::ATorpedoAUV() {
	PrimaryActorTick.bCanEverTick = true;

	// Set the default controller
	AIControllerClass = LoadClass<AController>(
		NULL, TEXT("/Script/Holodeck.TorpedoAUVController"), NULL, LOAD_None, NULL);
	AutoPossessAI = EAutoPossessAI::PlacedInWorld;

	this->MassInKG = 36;
	this->Volume = MassInKG / WaterDensity;	 // 0.0342867409204;
	this->CenterMass = FVector(0, 0, 0);	 // cm (unreal units)
	this->CenterBuoyancy = FVector(0, 0, 1); // cm (unreal units)
	// These values are completely guesstimations
	this->CoefficientOfDrag = 0.05;
	this->AreaOfDrag = 0.5;
}

void ATorpedoAUV::InitializeAgent() {
	RootMesh = Cast<UStaticMeshComponent>(RootComponent);

	Super::InitializeAgent();
}

// Called every frame when custom dynamics control scheme is active
void ATorpedoAUV::Tick(float DeltaSeconds) {
	Super::Tick(DeltaSeconds);

	// Convert linear acceleration to force
	FVector linAccel = FVector(CommandArray[0], CommandArray[1], CommandArray[2]);
	linAccel = ConvertLinearVector(linAccel, ClientToUE);

	// Convert angular acceleration to torque
	FVector angAccel = FVector(CommandArray[3], CommandArray[4], CommandArray[5]);
	angAccel = ConvertAngularVector(angAccel, NoScale);

	RootMesh->GetBodyInstance()->AddForce(linAccel, true, true);
	RootMesh->GetBodyInstance()->AddTorqueInRadians(angAccel, true, true);
}

/** Based on the models found in
 * Preliminary Evaluation of Cooperative Navigation of Underwater Vehicles
 * 		without a DVL Utilizing a Dynamic Process Model, Section III-3)
 * Control Inputs https://ieeexplore.ieee.org/document/8460970
 */
void ATorpedoAUV::ApplyFin(int i, float command) {
	// Get rotations
	float	 commandAngle = FMath::Clamp(command, TAUV_MIN_FIN, TAUV_MAX_FIN);
	FRotator bodyToWorld = this->GetActorRotation();
	FRotator finToBody = UKismetMathLibrary::ComposeRotators(
		FRotator(commandAngle, 0, 0),
		finRotation[i]); // Note: rotators use convention (pitch, yaw, roll). The
						 // command is a pitch.

	// get velocity at fin location, in fin frame
	FVector finWorld = RootMesh->GetCenterOfMass()
		+ bodyToWorld.RotateVector(finTranslation[i] - CenterMass);
	FVector velWorld = RootMesh->GetPhysicsLinearVelocityAtPoint(finWorld);
	// METERS/sec (unreal is dumb, distance is in cm/s but velocity
	// is in m/s)
	FVector velOceanCurrent = this->GetOceanCurrentVelocity();
	FVector relativeVel = velWorld - velOceanCurrent;
	FVector velBody = bodyToWorld.UnrotateVector(relativeVel);
	FVector velFin = finToBody.UnrotateVector(velBody);

	// get angle of flow relative to fin and transform to body frame
	double angle = UKismetMathLibrary::DegAtan2(-velFin.Z, velFin.X);
	while (angle - commandAngle > 90) {
		angle -= 180;
	}
	while (angle - commandAngle < -90) {
		angle += 180;
	}
	FRotator flowToBody = UKismetMathLibrary::ComposeRotators(
		FRotator(angle, 0, 0), finToBody); // finRotation[i]

	// Calculate lift and drag forces in the flow frame
	// Parameters are guesses, and equations are linearizations only valid to
	// angles of about 12 degrees... so this is all very approximate.
	double fin_area = 0.01; // m^2, guess for IVER3
	double Cl_a = 0.1;		// 2D lift coefficient per degree
	double AR = 1.2;		// aspect ratio
	double e = 0.8;			// Oswald efficiency factor
	double Cd0 = 0.01;		// zero-lift drag coefficient
	double u2 = velFin.Z * velFin.Z + velFin.X * velFin.X;
	// (m/s)^2, squared velocity of fin in flow frame
	double CL = (Cl_a / (1 + Cl_a / (3.14 * AR * e)))
		* (angle * 3.14 / 180);									 // lift coefficient
	double CD = Cd0 + CL * CL / (3.14 * AR * e);				 // drag coefficient
	double lift = 0.5 * this->WaterDensity * u2 * fin_area * CL; // force in N
	double drag = 0.5 * this->WaterDensity * u2 * fin_area * CD; // force in N

	double	scale = 0.12; // tuned to make forces reasonable
	FVector forceFlow = FVector(-drag, 0, lift) * scale;
	forceFlow = forceFlow.GetClampedToMaxSize(300); // clamp forces
	if (velBody.X < 0) {
		forceFlow *= -1;
	} // flip it if we're going backwards

	// Move force into world frame & apply at fin location
	FVector forceBody = flowToBody.RotateVector(forceFlow);
	FVector forceWorld = bodyToWorld.RotateVector(forceBody);
	if (RootMesh->GetCenterOfMass().Z
		<= 0) { // simple check to make sure we're underwater, could be made more
		// robust
		RootMesh->AddForceAtLocation(forceWorld, finWorld);
	}

	// Draw Debug Lines
	// UE_LOG(LogHolodeck, Warning, TEXT("Angle: %f, lift %f, drag %f"), angle,
	// forceFlow.Z, -forceFlow.X); UE_LOG(LogHolodeck, Warning, TEXT("Angle: %f,
	// lift %f, drag %f"), angle, lift, -drag); FTransform finCoord =
	// FTransform(finToBody, finTranslation[i]) * GetActorTransform();
	// DrawDebugLine(GetWorld(), finWorld, finWorld+forceWorld, FColor::Red,
	// false, .1, ECC_WorldStatic, 1.f); DrawDebugCoordinateSystem(GetWorld(),
	// finCoord.GetTranslation(), finCoord.Rotator(), 15, false, .1,
	// ECC_WorldStatic, 1.f);
}

void ATorpedoAUV::ApplyThrust(float thrust) {
	float ThrustToApply = FMath::Clamp(thrust, TAUV_MIN_THRUST, TAUV_MAX_THRUST);

	FRotator bodyToWorld = this->GetActorRotation();
	FVector	 COM = RootMesh->GetCenterOfMass();

	FVector Thrust = FVector(ThrustToApply, 0, 0);
	Thrust = ConvertLinearVector(Thrust, ClientToUE);
	Thrust = bodyToWorld.RotateVector(Thrust);
	FVector ThrusterWorld =
		RootMesh->GetCenterOfMass() + bodyToWorld.RotateVector(thruster - CenterMass);
	// get thruster location in global frame
	if (RootMesh->GetCenterOfMass().Z
		<= 0) { // simple check to make sure we're underwater, could be made more
		// robust
		RootMesh->AddForceAtLocation(Thrust, ThrusterWorld);
	}

	// Draw Debug Line
	// DrawDebugLine(GetWorld(), COM + bodyToWorld.RotateVector(thruster), COM +
	// bodyToWorld.RotateVector(thruster) + Thrust*0.25, FColor::Red, false, .1,
	// ECC_WorldStatic, 2.f);
}

// For empty dynamics, damping is disabled
// Enable it when using thrusters & fins
void ATorpedoAUV::EnableDamping() {
	RootMesh->SetLinearDamping(0.75);
	RootMesh->SetAngularDamping(0.5);
}