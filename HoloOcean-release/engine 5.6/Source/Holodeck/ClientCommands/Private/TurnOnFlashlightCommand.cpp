// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "TurnOnFlashlightCommand.h"
#include "FlashlightManager.h"
#include "Holodeck.h"
#include "HolodeckBuoyantAgent.h"
#include "HolodeckGameMode.h"

void UTurnOnFlashlightCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("UTurnOnFlashlightCommand::Execute"));

	if (StringParams.size() != 1 || NumberParams.size() != 10) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in UTurnOnFlashlightCommand. "
				"Command not executed."));
		return;
	}

	AHolodeckGameMode* GameTarget = static_cast<AHolodeckGameMode*>(Target);
	if (GameTarget == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UCommand::Target is not a UHolodeckGameMode*. "
				"UTurnOnFlashlightCommand::Flashlight not turned on."));
		return;
	}

	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("UTurnOnFlashlightCommand::Execute found world as nullptr."));
		return;
	}

	// Flashlight arguments
	FString	 FlashlightName = FString(StringParams[0].c_str()).ToLower();
	float	 lightIntensity = FMath::Clamp(NumberParams[0], 0.0f, 100000.0f);
	float	 beamWidth = FMath::Clamp(NumberParams[1], 1.0f, 80.0f);
	FVector	 lightLocation = FVector(NumberParams[2], NumberParams[3], NumberParams[4]);
	FRotator lightAngle = FRotator(
		FMath::Clamp(NumberParams[5], -70.0f, 70.0f),
		FMath::Clamp(NumberParams[6], -70.0f, 70.0f),
		0);
	FLinearColor lightColor = FLinearColor(
		FMath::Clamp(NumberParams[7], 0.0f, 1.0f),
		FMath::Clamp(NumberParams[8], 0.0f, 1.0f),
		FMath::Clamp(NumberParams[9], 0.0f, 1.0f));

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
		Manager->SetBeamWidth(beamWidth);
		Manager->SetLightPosition(lightLocation);
		Manager->SetLightAngle(lightAngle);
		Manager->SetLightColor(lightColor);
		UE_LOG(LogHolodeck, Log, TEXT("Flashlight turned ON."));
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