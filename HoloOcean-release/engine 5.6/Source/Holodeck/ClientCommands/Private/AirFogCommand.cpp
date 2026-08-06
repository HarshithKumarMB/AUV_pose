// MIT License (c) 2019 BYU PCCL see LICENSE file
#include "AirFogCommand.h"
#include "Holodeck.h"
#include "HolodeckGameMode.h"

#include "Components/PostProcessComponent.h"
#include "Engine/PostProcessVolume.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"

void UAirFogCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("UAirFogCommand::Execute fog"));
	if (NumberParams.size() != 5) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in UAirFogCommand. Command "
				"not executed."));
		return;
	}
	AHolodeckGameMode* GameTarget = static_cast<AHolodeckGameMode*>(Target);
	if (GameTarget == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UCommand::Target is not a UHolodeckGameMode*. "
				"UAirFogCommand::Fog not changed."));
		return;
	}
	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UAirFogCommand::Execute found world as nullptr. Fog not "
				"changed."));
		return;
	}

	// All Post Process Volumes
	TArray<AActor*> FoundPPVolumes;
	UGameplayStatics::GetAllActorsOfClass(
		World, APostProcessVolume::StaticClass(), FoundPPVolumes);
	if (FoundPPVolumes.Num() == 0) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("No Post Process Volumes found in the scene. Fog not changed."));
		return;
	}

	// Read arguments
	float fogDensity = FMath::Clamp(NumberParams[0], 0.0f, 10.0f);
	float fogDepth = (FMath::Clamp(NumberParams[1], 0.0f, 10.0f))
		* 40000; // depth is in the range of 40k to 400k, but user inputs 1 to 10
	FLinearColor fogColor = FLinearColor(
		FMath::Clamp(NumberParams[2], 0.0f, 1.0f),
		FMath::Clamp(NumberParams[3], 0.0f, 1.0f),
		FMath::Clamp(NumberParams[4], 0.0f, 1.0f),
		1 // alpha set to 1
	);

	// Getting water PPV based on tag
	APostProcessVolume* AirPPV = nullptr;
	for (AActor* actor : FoundPPVolumes) {
		if (actor->Tags.Contains("AirPPV")) {
			AirPPV = Cast<APostProcessVolume>(actor);
			break;
		}
	}
	if (!AirPPV) {
		UE_LOG(LogHolodeck, Warning, TEXT("No 'AirPPV' found. Fog not changed."));
		return;
	}

	// Ensure blend is fully applied
	AirPPV->BlendWeight = 1.0f;
	if (fogDensity == 0) {
		AirPPV->BlendWeight = 0.0f;
	}

	// Access the post process settings
	FPostProcessSettings&		PPSettings = AirPPV->Settings;
	TArray<FWeightedBlendable>& Blendables = PPSettings.WeightedBlendables.Array;

	for (int32 i = 0; i < Blendables.Num(); ++i) {
		UObject*			Obj = Blendables[i].Object;
		UMaterialInterface* BaseMat = Cast<UMaterialInterface>(Obj);
		if (BaseMat && !Cast<UMaterialInstanceDynamic>(BaseMat)) {
			// Create dynamic instance
			UMaterialInstanceDynamic* DynMat =
				UMaterialInstanceDynamic::Create(BaseMat, this);
			if (DynMat) {
				Blendables[i].Object = DynMat; // Replace in the blendables
				UE_LOG(LogHolodeck, Log, TEXT("Converted to dynamic material."));

				// Now set parameters
				DynMat->SetScalarParameterValue(FName("Fog_Depth"), fogDepth);
				DynMat->SetScalarParameterValue(FName("Fog_Opacity"), fogDensity);
				DynMat->SetScalarParameterValue(FName("Fog_Transition"), 0.1);
				DynMat->SetVectorParameterValue(FName("Fog_Color"), fogColor);
			}
		} else if (
			UMaterialInstanceDynamic* DynMat = Cast<UMaterialInstanceDynamic>(Obj)) {
			// Already dynamic, just set params
			DynMat->SetScalarParameterValue(FName("Fog_Depth"), fogDepth);
			DynMat->SetScalarParameterValue(FName("Fog_Opacity"), fogDensity);
			DynMat->SetScalarParameterValue(FName("Fog_Transition"), 0.1);
			DynMat->SetVectorParameterValue(FName("Fog_Color"), fogColor);
		}
	}
}