// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "SetRainParametersCommand.h"
#include "Holodeck.h"
#include "HolodeckGameMode.h"
#include "WeatherManager.h"

void USetRainParametersCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("USetRainParametersCommand::Execute"));

	if (NumberParams.size() != 4) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in "
				"USetRainParametersCommand. Command not executed."));
		return;
	}

	AHolodeckGameMode* GameTarget = static_cast<AHolodeckGameMode*>(Target);
	if (GameTarget == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UCommand::Target is not a UHolodeckGameMode*. "
				"USetRainParametersCommand::Rain not modified."));
		return;
	}

	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"USetRainParametersCommand::Execute found world as nullptr. "
				"Rain not modified."));
		return;
	}

	FVector newRainVelocity =
		FVector(NumberParams[0], NumberParams[1], NumberParams[2]);
	float newSpawnRate = NumberParams[3];

	// Find WeatherManager in the world and set variables
	for (TActorIterator<AWeatherManager> It(GameTarget->GetWorld()); It; ++It) {
		AWeatherManager* Manager = *It;
		if (Manager) {
			Manager->SetRainVelocity(newRainVelocity);
			Manager->SetSpawnRate(newSpawnRate);
			break;
		}
	}
}