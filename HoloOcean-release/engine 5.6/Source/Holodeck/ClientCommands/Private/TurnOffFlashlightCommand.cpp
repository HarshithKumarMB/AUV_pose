// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "TurnOffFlashlightCommand.h"
#include "FlashlightManager.h"
#include "Holodeck.h"
#include "HolodeckBuoyantAgent.h"
#include "HolodeckGameMode.h"

void UTurnOffFlashlightCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("UTurnOffFlashlightCommand::Execute"));

	if (StringParams.size() != 1 || NumberParams.size() != 0) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in "
				"UTurnOffFlashlightCommand. Command not executed."));
		return;
	}

	AHolodeckGameMode* GameTarget = static_cast<AHolodeckGameMode*>(Target);
	if (GameTarget == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UCommand::Target is not a UHolodeckGameMode*. "
				"UTurnOffFlashlightCommand::Flashlight not turned off."));
		return;
	}

	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("UTurnOffFlashlightCommand::Execute found world as nullptr."));
		return;
	}

	// Flashlight arguments
	FString FlashlightName = FString(StringParams[0].c_str()).ToLower();
	float	lightIntensity = 0.0;

	// Turning flashlight name to number
	int lightNumber = 0;

	if (FlashlightName.Equals("flashlight1")) {
		lightNumber = 1;
	} else if (FlashlightName.Equals("flashlight2")) {
		lightNumber = 2;
	} else if (FlashlightName.Equals("flashlight3")) {
		lightNumber = 3;
	} else if (FlashlightName.Equals("flashlight4")) {
		lightNumber = 4;
	} else {
		UE_LOG(
			LogHolodeck, Error, TEXT("Unknown flashlight name: %s"), *FlashlightName);
		return;
	}

	// Flashlight settings
	for (TActorIterator<AFlashlightManager> It(GameTarget->GetWorld()); It; ++It) {
		AFlashlightManager* Manager = *It;
		if (!Manager)
			continue;

		Manager->SetLightNumber(lightNumber);
		Manager->SetLightIntensity(lightIntensity);

		UE_LOG(LogHolodeck, Log, TEXT("Flashlight turned OFF"));
	}

	for (TActorIterator<AHolodeckBuoyantAgent> It(GameTarget->GetWorld()); It; ++It) {
		AHolodeckBuoyantAgent* Vehicle = *It;
		if (!Vehicle)
			continue;

		// Call the Blueprint event (ApplyFlashlightSettingsFromManager)
		Vehicle->CallFunctionByNameWithArguments(
			TEXT("ApplyFlashlightSettingsFromManager"), *GLog, nullptr, true);
	}
}
