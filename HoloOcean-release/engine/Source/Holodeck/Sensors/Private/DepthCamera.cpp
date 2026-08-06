#include "DepthCamera.h"

#include "Engine.h"

#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Kismet/GameplayStatics.h"
#include "ShowFlags.h"

#include "Materials/Material.h"

#include "RHICommandList.h"

#include "Modules/ModuleManager.h"

UDepthCamera::UDepthCamera() {
	SensorName = "DepthCamera";
	Parent = nullptr;
}

void UDepthCamera::ParseSensorParms(FString ParmsJson) {
	Super::ParseSensorParms(ParmsJson);

	TSharedPtr<FJsonObject>		   JsonParsed;
	TSharedRef<TJsonReader<TCHAR>> JsonReader =
		TJsonReaderFactory<TCHAR>::Create(ParmsJson);
	if (FJsonSerializer::Deserialize(JsonReader, JsonParsed)) {

		if (JsonParsed->HasTypedField<EJson::Number>("TicksPerCapture")) {
			TicksPerCapture = JsonParsed->GetIntegerField("TicksPerCapture");
		}
		if (JsonParsed->TryGetNumberField("FovAngle", FovAngle)) {
			SetFOVAngle(FovAngle);
		}
		if (JsonParsed->HasTypedField<EJson::Number>("CaptureWidth")) {
			CaptureWidth = JsonParsed->GetIntegerField("CaptureWidth");
		}

		if (JsonParsed->HasTypedField<EJson::Number>("CaptureHeight")) {
			CaptureHeight = JsonParsed->GetIntegerField("CaptureHeight");
		}
	} else {
		UE_LOG(
			LogHolodeck,
			Fatal,
			TEXT("UShaderBasedSensor::ParseSensorParms:: Unable to parse json."));
	}
}

void UDepthCamera::InitializeSensor() {
	Super::InitializeSensor();
	Parent = this->GetAttachmentRootActor();

	// TargetTexture->SRGB = false; //No alpha
	// TargetTexture->CompressionSettings = TC_VectorDisplacementmap;
	// TargetTexture->RenderTargetFormat = RTF_RGBA8;
	// SceneCapture->PostProcessSettings.bOverride_AutoExposureBias = 1;

	// SceneCapture->ProjectionType = ECameraProjectionMode::Perspective;
	// SceneCapture->CompositeMode = SCCM_Overwrite;
	// SceneCapture->CaptureSource = SCS_SceneColorHDR;
	// SceneCapture->TextureTarget = TargetTexture;

	SceneCapture->CaptureSource = SCS_FinalColorHDR;
	SceneCapture->MaxViewDistanceOverride = 100000.0f; // 1 km
	TargetTexture->InitCustomFormat(CaptureWidth, CaptureHeight, PF_R8G8B8A8, true);
	TargetTexture->UpdateResource();
	SceneCapture->TextureTarget = TargetTexture;

	UMaterialInterface* DepthMat = Cast<UMaterialInterface>(StaticLoadObject(
		UMaterialInterface::StaticClass(),
		nullptr,
		TEXT(
			"Material'/Game/PostProcessingMaterials/DepthEffectMaterial_GLSL.DepthEffectMaterial_GLSL'")));

	if (DepthMat) {
		// Option A (works if your engine accepts UMaterialInterface as a blendable):
		// SceneCapture->PostProcessSettings.AddBlendable(DepthMat, 1.0f);

		// Option B (most robust across UE versions):
		SceneCapture->PostProcessSettings.WeightedBlendables.Array.Add(
			FWeightedBlendable(1.0f, DepthMat));
	} else {
		StaticLoadObject(
			UMaterialInterface::StaticClass(),
			nullptr,
			TEXT(
				"Material'/Game/PostProcessingMaterials/DepthEffectMaterial.DepthEffectMaterial'"));
		if (DepthMat) {
			SceneCapture->PostProcessSettings.WeightedBlendables.Array.Add(
				FWeightedBlendable(1.0f, DepthMat));
		} else {
			UE_LOG(LogTemp, Error, TEXT("Failed to load DepthEffectMaterial!"));
		}
	}
}

void UDepthCamera::TickSensorComponent(
	float						 DeltaTime,
	ELevelTick					 TickType,
	FActorComponentTickFunction* ThisTickFunction) {

	TickCounter++;
	if (TickCounter >= TicksPerCapture) {
		RenderRequest.RetrievePixels(Buffer, TargetTexture);

		TickCounter -= TicksPerCapture;
	}
}
