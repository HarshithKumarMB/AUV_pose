// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "AdjustTPSCommand.h"
#include "Holodeck.h"
#include "HolodeckGameMode.h"
#include "HolodeckWorldSettings.h"

void UAdjustTPSCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("UAdjustTPSCommand::Execute adjust ticks per sec"));

	if (StringParams.size() != 0 || NumberParams.size() != 1) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in UAdjustTPSCommand. "
				"Command not executed."));
		return;
	}

	// Get and validate World
	UWorld* World = GEngine->GetWorldContexts()[0].World();
	if (!World) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"UAdjustTPSCommand::Execute: World not found. Command not "
				"executed."));
		return;
	}

	// Get and validate WorldSettings
	AWorldSettings* BaseWorldSettings = World->GetWorldSettings();
	if (!BaseWorldSettings) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"UAdjustTPSCommand::Execute: Base WorldSettings not found. "
				"Command not executed."));
		return;
	}

	AHolodeckWorldSettings* WorldSettings =
		Cast<AHolodeckWorldSettings>(BaseWorldSettings);
	if (!WorldSettings) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"UAdjustTPSCommand::Execute: WorldSettings is not of type "
				"AHolodeckWorldSettings. Found type: %s. Command not executed."),
			*BaseWorldSettings->GetClass()->GetName());
		return;
	}

	// Validate TPS value
	float TicksPerSec = NumberParams[0];
	if (TicksPerSec <= 0) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"UAdjustTPSCommand::Execute: Ticks per second must be greater "
				"than 0, got %f. Command not executed."),
			TicksPerSec);
		return;
	}

	// Apply the change
	float OldDelta = WorldSettings->GetConstantTimeDeltaBetweenTicks();
	WorldSettings->SetConstantTimeDeltaBetweenTicks(1.0 / TicksPerSec);
	float NewDelta = WorldSettings->GetConstantTimeDeltaBetweenTicks();

	UE_LOG(
		LogHolodeck,
		Log,
		TEXT(
			"UAdjustTPSCommand::Execute: Successfully changed TPS from %f to "
			"%f (delta time from %f to %f)"),
		1.0 / OldDelta,
		TicksPerSec,
		OldDelta,
		NewDelta);
}
