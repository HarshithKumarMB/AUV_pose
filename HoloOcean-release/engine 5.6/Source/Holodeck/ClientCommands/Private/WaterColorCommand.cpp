#include "WaterColorCommand.h"
#include "Holodeck.h"
#include "HolodeckGameMode.h"

void UWaterColorCommand::Execute() {
	// UE_LOG(LogHolodeck, Log, TEXT("UWaterColorCommand::Change Water Color"));
	if (NumberParams.size() != 3) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in UWaterColorCommand. "
				"Command not executed."));
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
			TEXT("UAddSensorCommand::Execute found world as nullptr."));
		return;
	}

	TArray<AActor*> actors;
	UGameplayStatics::GetAllActorsOfClass(
		World, APostProcessVolume::StaticClass(), actors);

	for (AActor* actor : actors) {
		if (actor->Tags.Contains("AirPPV")) {
			continue;
		}
		APostProcessVolume* ppv = Cast<APostProcessVolume>(actor);
		FLinearColor		tint;
		tint.R = NumberParams[0];
		tint.G = NumberParams[1];
		tint.B = NumberParams[2];
		tint.A = 1; // this value doesn't affect the water color -- but still needs
					// to be set
		ppv->Settings.SceneColorTint = tint;
	}
}
