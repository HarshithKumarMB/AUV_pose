// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "HolodeckBuoyantAgent.h"
#include "Holodeck.h"
#include "HolodeckAgent.h"
#include "VectorField/VectorFieldStatic.h"

void AHolodeckBuoyantAgent::InitializeAgent() {
	Super::InitializeAgent();

	// Get GravityVectority from world
	AWorldSettings* WorldSettings = GetWorld()->GetWorldSettings(false, false);
	Gravity = WorldSettings->GetGravityZ()
		/ -100; // this converts gravity from cm/s^2 (unreal) to m/s^2

	// Set Mass
	RootMesh->SetMassOverrideInKg("", MassInKG);
	RootMesh->SetCenterOfMass(CenterMass); // set the center of mass in the body frame

	// Set Bounding Box (if it hasn't been set by hand)
	if (BoundingBox.GetExtent() == FVector(0, 0, 0))
		BoundingBox = RootMesh->GetStaticMesh()->GetBoundingBox();

	// Sample points (if they haven't already been set)
	if (SurfacePoints.Num() == 0) {
		for (int i = 0; i < NumSurfacePoints; i++) {
			FVector random = UKismetMathLibrary::RandomPointInBoundingBox(
				FVector(0, 0, 0), BoundingBox.GetExtent());
			SurfacePoints.Add(random);
		}
	}
	// Otherwise make sure our count is correct (we'll use it later)
	else {
		NumSurfacePoints = SurfacePoints.Num();
	}
}

void AHolodeckBuoyantAgent::Tick(float DeltaSeconds) {
	Super::Tick(DeltaSeconds);
	if (octreeGlobal != nullptr)
		updateOctree(octreeLocal, octreeGlobal);
}

void AHolodeckBuoyantAgent::BeginDestroy() {
	Super::BeginDestroy();

	if (octreeLocal != nullptr)
		delete octreeLocal;
	if (octreeGlobal != nullptr)
		delete octreeGlobal;
}

void AHolodeckBuoyantAgent::ApplyBuoyancyDragForce() {
	// Get all the values we need once
	FVector ActorLocation = GetActorLocation(); // transformation from body to global UE
												// frame, expressed in global UE frame
	FRotator ActorRotation =
		GetActorRotation();					   // rotation from body to global UE frame
	FVector COM = RootMesh->GetCenterOfMass(); // COM in global UE frame
	FVector COB = COM
		+ ActorRotation.RotateVector(
			CenterBuoyancy - CenterMass); // COB in global frame

	FVector* points = SurfacePoints.GetData();
	int		 count = 0;
	for (int i = 0; i < NumSurfacePoints; i++) {
		FVector p_world = ActorLocation + ActorRotation.RotateVector(points[i]);
		if (p_world.Z < SurfaceLevel)
			count++;
	}
	float ratio = count * 1.0 / NumSurfacePoints;

	// Gravity
	FVector GravityVector = FVector(0, 0, -Gravity * MassInKG); // Newtons
	GravityVector = ConvertLinearVector(
		GravityVector, ClientToUE); // converts Newtons to centiNewtons
	RootMesh->AddForceAtLocation(GravityVector, COM);

	// Buoyant Force (Applied at COM)
	float	BuoyantForce = Volume * Gravity * WaterDensity * ratio;
	FVector BuoyantVector = FVector(0, 0, BuoyantForce); // Newtons, in global frame
	BuoyantVector = ConvertLinearVector(
		BuoyantVector, ClientToUE); // converts Newtons to centiNewtons
	RootMesh->AddForceAtLocation(BuoyantVector, COB);

	// Drag Force (Applied at COM)
	FVector OceanCurrentsVel = GetOceanCurrentVelocity();
	FVector AUVVel =
		RootMesh->GetBodyInstance()->GetUnrealWorldVelocity() / 100.0; // in m/s
	FVector RelativeVel = AUVVel - OceanCurrentsVel;
	FVector DragForce = -0.5 * WaterDensity * RelativeVel.SizeSquared()
		* CoefficientOfDrag * AreaOfDrag * RelativeVel.GetSafeNormal();
	RootMesh->AddForceAtLocation(
		DragForce * ratio,
		COM); // Apply drag force proportional to how submerged the vehicle is
			  // (this is an approximation, as drag coeff is not constant with
			  // depth underwater)

	// Draw Debug Lines
	if (Ocean_Current_Vehicle_Debugging) {
		DrawDebugLine(
			GetWorld(),
			COM,
			COM + GravityVector * 0.02,
			FColor::Green,
			false,
			0,
			ECC_WorldStatic,
			2.f);
		DrawDebugLine(
			GetWorld(),
			COB,
			COB + BuoyantVector * 0.02,
			FColor::Blue,
			false,
			0,
			ECC_WorldStatic,
			2.f);
		DrawDebugLine(
			GetWorld(),
			COM,
			COM + OceanCurrentsVel * 0.02,
			FColor::Red,
			false,
			0,
			ECC_WorldStatic,
			2.f);
	}
}

FVector AHolodeckBuoyantAgent::GetOceanCurrentVelocity() {
	FVector OceanCurrentsVel = FVector(0, 0, 0);
	OceanCurrentsVel[0] = Ocean_Current_Velocity_X;
	OceanCurrentsVel[1] = -Ocean_Current_Velocity_Y;
	OceanCurrentsVel[2] = Ocean_Current_Velocity_Z;
	return OceanCurrentsVel;
}

void AHolodeckBuoyantAgent::ShowBoundingBox(float DeltaTime) {
	FVector location = GetActorLocation();
	DrawDebugBox(
		GetWorld(),
		location,
		BoundingBox.GetExtent(),
		GetActorQuat(),
		FColor::Red,
		false,
		DeltaTime,
		0,
		1);
}

void AHolodeckBuoyantAgent::ShowSurfacePoints(float DeltaTime) {
	FVector	 ActorLocation = GetActorLocation();
	FRotator ActorRotation = GetActorRotation();
	FVector* points = SurfacePoints.GetData();

	for (int i = 0; i < NumSurfacePoints; i++) {
		FVector p_world = ActorLocation + ActorRotation.RotateVector(points[i]);
		DrawDebugPoint(GetWorld(), p_world, 5, FColor::Red, false, DeltaTime);
	}
}

Octree* AHolodeckBuoyantAgent::makeOctree() {
	if (octreeGlobal == nullptr) {
		UE_LOG(LogHolodeck, Log, TEXT("HolodeckBuoyantAgent::Making Octree"));
		float OctreeMin = Octree::OctreeMin;
		float OctreeMax = Octree::OctreeMin;

		// Shrink to the smallest cube the actor fits in
		FVector center = BoundingBox.GetCenter() + GetActorLocation();
		float	extent = BoundingBox.GetExtent().GetAbsMax() * 2;
		while (OctreeMax < extent) {
			OctreeMax *= 2;
		}

		// Otherwise, make the octrees
		octreeGlobal = Octree::makeOctree(center, OctreeMax, OctreeMin, GetName());
		if (octreeGlobal) {
			octreeGlobal->isAgent = true;
			octreeGlobal->file = "AGENT";

			// Convert our global octree to a local one
			octreeLocal = cleanOctree(octreeGlobal);
		} else {
			UE_LOG(
				LogHolodeck,
				Warning,
				TEXT("HolodeckBuoyantAgent:: Failed to make Octree"));
		}
	}

	return octreeGlobal;
}

Octree* AHolodeckBuoyantAgent::cleanOctree(Octree* globalFrame) {
	Octree* local = new Octree;
	local->loc =
		GetActorRotation().UnrotateVector(globalFrame->loc - GetActorLocation());
	local->normal = GetActorRotation().UnrotateVector(globalFrame->normal);

	for (Octree* tree : globalFrame->leaves) {
		Octree* l = cleanOctree(tree);
		local->leaves.Add(l);
	}

	return local;
}

void AHolodeckBuoyantAgent::updateOctree(Octree* localFrame, Octree* globalFrame) {
	globalFrame->loc =
		GetActorLocation() + GetActorRotation().RotateVector(localFrame->loc);
	globalFrame->normal = GetActorRotation().RotateVector(localFrame->normal);

	for (int i = 0; i < globalFrame->leaves.Num(); i++) {
		updateOctree(localFrame->leaves[i], globalFrame->leaves[i]);
	}
}