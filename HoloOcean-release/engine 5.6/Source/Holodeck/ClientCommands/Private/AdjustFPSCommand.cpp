// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "AdjustFPSCommand.h"
#include "Holodeck.h"
#include "HolodeckGameMode.h"

void UAdjustFPSCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("UAdjustFPSCommand::Execute adjust fps"));

	if (StringParams.size() != 0 || NumberParams.size() != 1) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length in UAdjustFPSCommand. Command not "
				"executed."));
		return;
	}

	FString command = "t.MaxFPS " + FString::FromInt(NumberParams[0]);

	UWorld* World = GEngine->GetWorldContexts()[0].World();
	if (World && World->GetFirstPlayerController()) {
		APlayerController* PC = World->GetFirstPlayerController();
		PC->ConsoleCommand(*command, true);
		UE_LOG(
			LogHolodeck,
			Log,
			TEXT("Executed command '%s'. FPS set to %d."),
			*command,
			NumberParams[0]);
	} else {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT("World or PlayerController not found. Command '%s' not executed."),
			*command);
	}
}
