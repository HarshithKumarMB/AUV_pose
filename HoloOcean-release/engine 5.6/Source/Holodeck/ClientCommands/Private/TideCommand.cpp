// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "TideCommand.h"
#include "Holodeck.h"
#include "HolodeckBuoyantAgent.h"
#include "HolodeckGameMode.h"

// Define static members
FVector UTideCommand::PPVUnderwater = FVector::ZeroVector;
FVector UTideCommand::PPVAir = FVector::ZeroVector;
bool	UTideCommand::waterppvfirst = true;
bool	UTideCommand::airppvfirst = true;

void UTideCommand::Execute() {
	// UE_LOG(LogHolodeck, Log, TEXT("UTideCommand::Execute tide"));

	if (NumberParams.size() != 2) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in UTideCommand. Command not "
				"executed."));
		return;
	}

	AHolodeckGameMode* GameTarget = static_cast<AHolodeckGameMode*>(Target);
	if (GameTarget == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("UCommand::Target is not a UHolodeckGameMode*."));
		return;
	}

	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("UTideCommand::Execute found world as nullptr."));
		return;
	}

	TArray<AActor*> found_agents; // this gets our robots
	UGameplayStatics::GetAllActorsOfClass(
		World, AHolodeckAgent::StaticClass(), found_agents);

	TArray<AActor*> found_volumes; // this gets our post process volume
	UGameplayStatics::GetAllActorsOfClass(
		World, APostProcessVolume::StaticClass(), found_volumes);

	TArray<AActor*> found_actors; // this grabs every actor
	UGameplayStatics::GetAllActorsOfClass(World, AActor::StaticClass(), found_actors);

	float			adjustment = NumberParams[0] * 100;
	// this filters and sorts out the actors that we want
	TArray<AActor*> found_water;
	TArray<AActor*> found_floats;
	for (AActor* actor : found_actors) {
		bool found = 0;
		if (actor->Tags.Contains("WaterSurface")) {
			found_water.Add(actor);
		} else if (actor->Tags.Contains("Float")) {
			found_floats.Add(actor);
		}
		// else if (actor->GetActorLabel() == "PhysicsVolume") { -- this makes it
		// really laggy -- also getactorlabel doesn't work with packaging -- have to
		// either have a specific type or gotta have tags found_volumes.Add(actor);
		// }
	}
	// moving the water plane
	FVector water_level;
	for (int32 i = 0; i < found_water.Num(); i++) {
		water_level = found_water[i]->GetActorLocation();
		if (!NumberParams[1]) {
			water_level.Z += adjustment;
			found_water[i]->SetActorLocation(water_level);
		} else {
			water_level.Z = adjustment;
			found_water[i]->SetActorLocation(water_level);
		}
	}

	// moving the floatable -- using collision checking
	for (AActor* actor : found_floats) {
		FVector actor_location = actor->GetActorLocation();
		if (!NumberParams[1]) {
			actor_location.Z += adjustment;
		} else {
			actor_location.Z = adjustment;
		}
		FHitResult hit;
		if (actor_location.Z > water_level.Z) {
			actor_location.Z = water_level.Z;
			actor->SetActorLocation(actor_location, true, &hit);
		} else {
			actor->SetActorLocation(actor_location, true, &hit);
		}

		// if (hit.IsValidBlockingHit()) {} -- could be useful if want to do
		// something when the object hit the ground eg. explode
	}

	// moving the volumes
	for (int32 i = 0; i < found_volumes.Num(); i++) {
		FVector volume_location = found_volumes[i]->GetActorLocation();
		// UE_LOG(LogHolodeck, Log, TEXT("First: %i"), waterppvfirst);
		if (waterppvfirst) {
			if (found_volumes[i]->Tags.Contains("WaterPPV")) {
				PPVUnderwater = found_volumes[i]->GetActorLocation();
				waterppvfirst = false;
			}
		}
		if (airppvfirst) {
			if (found_volumes[i]->Tags.Contains("AirPPV")) {
				PPVAir = found_volumes[i]->GetActorLocation();
				airppvfirst = false;
			}
		}
		// UE_LOG(LogHolodeck, Log, TEXT("Last: %i"), waterppvfirst);
		// UE_LOG(LogHolodeck, Log, TEXT("Post Process volume befroe tide: %f, %f,
		// %f"), volume_location.X, volume_location.Y, volume_location.Z);
		if (!NumberParams[1]) {
			volume_location.Z += adjustment;
			found_volumes[i]->SetActorLocation(volume_location);
		} else {
			if (found_volumes[i]->Tags.Contains("WaterPPV")) {
				// UE_LOG(LogHolodeck, Log, TEXT("Initial ppv, %f,%f,%f"),
				// PPVUnderwater.X, PPVUnderwater.Y, PPVUnderwater.Z);
				FVector WaterLocation(
					PPVUnderwater.X, PPVUnderwater.Y, PPVUnderwater.Z + adjustment);
				// UE_LOG(LogHolodeck, Log, TEXT("final ppv, %f,%f,%f"),
				// PPVUnderwater.X, PPVUnderwater.Y, PPVUnderwater.Z);
				// UE_LOG(LogHolodeck, Log, TEXT("Initial water, %f,%f,%f"),
				// found_volumes[i]->GetActorLocation().X,
				// found_volumes[i]->GetActorLocation().Y,
				// found_volumes[i]->GetActorLocation().Z);
				found_volumes[i]->SetActorLocation(WaterLocation);
				// UE_LOG(LogHolodeck, Log, TEXT("Final water, %f,%f,%f"),
				// found_volumes[i]->GetActorLocation().X,
				// found_volumes[i]->GetActorLocation().Y,
				// found_volumes[i]->GetActorLocation().Z);
			}
			if (found_volumes[i]->Tags.Contains("AirPPV")) {
				FVector AirLocation(PPVAir.X, PPVAir.Y, PPVAir.Z + adjustment);
				found_volumes[i]->SetActorLocation(AirLocation);
			}
		}

		// UE_LOG(LogHolodeck, Log, TEXT("Post Process volume after tide: %f, %f,
		// %f"), volume_location.X, volume_location.Y, volume_location.Z);
	}

	// moving the agents
	for (int32 i = 0; i < found_agents.Num(); i++) {
		AHolodeckBuoyantAgent* BuoyantAgent =
			Cast<AHolodeckBuoyantAgent>(found_agents[i]);
		if (i == 0) {
			// UE_LOG(LogHolodeck, Log, TEXT("adjustment %f, surface level %f"),
			// adjustment, BuoyantAgent->SurfaceLevel); UE_LOG(LogHolodeck, Log,
			// TEXT("new surface level %f"), BuoyantAgent->SurfaceLevel);
		}
		if (!NumberParams[1]) {
			BuoyantAgent->SurfaceLevel += adjustment;
		} else {
			BuoyantAgent->SurfaceLevel = adjustment;
		}
	}
}
