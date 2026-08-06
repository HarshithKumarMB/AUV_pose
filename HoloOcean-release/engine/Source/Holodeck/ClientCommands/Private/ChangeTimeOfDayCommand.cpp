// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "ChangeTimeOfDayCommand.h"
#include "Holodeck.h"
#include "HolodeckGameMode.h"
#include "Math/UnrealMathUtility.h"

#include "Components/DirectionalLightComponent.h"
#include "Engine/DirectionalLight.h"

void UChangeTimeOfDayCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("UChangeTimeOfDayCommand::Execute"));

	if (NumberParams.size() != 1) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in UChangeTimeOfDayCommand. "
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
				"UChangeTimeOfDayCommand::Time of day not changed."));
		return;
	}

	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UChangeTimeOfDayCommand::Execute found world as nullptr. Time "
				"of day not changed."));
		return;
	}

	TArray<AActor*> DirectionalLights;
	UGameplayStatics::GetAllActorsOfClass(
		World, ADirectionalLight::StaticClass(), DirectionalLights);

	if (DirectionalLights.Num() == 0) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"No Directional Lights found in the scene. Time of day not "
				"changed."));
		return;
	}

	float TimeOfDay = NumberParams[0];
	// Validate input
	if (TimeOfDay < 0 || TimeOfDay > 24) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("Invalid time of day: %f. Must be in [0, 24]."),
			TimeOfDay);
		return;
	}

	// Compute pitch: 0 = midnight = 90°, 12 = noon = 270°
	float Pitch = FMath::Fmod(90.0f + (TimeOfDay * 180.0f / 12.0f), 360.0f);

	// Changing the light rotation parameter
	for (int32 j = 0; j < DirectionalLights.Num(); j++) {
		ADirectionalLight* Light = Cast<ADirectionalLight>(DirectionalLights[j]);
		ULightComponent*   BaseLightComponent = Light->GetLightComponent();
		UDirectionalLightComponent* LightComponent =
			Cast<UDirectionalLightComponent>(BaseLightComponent);

		FRotator LightRotation = FRotator::ZeroRotator;
		LightRotation.Yaw = 90.0f;
		LightRotation.Roll = 0.0f;
		LightRotation.Pitch = Pitch;

		LightComponent->SetRelativeRotation(LightRotation);
	}
}
