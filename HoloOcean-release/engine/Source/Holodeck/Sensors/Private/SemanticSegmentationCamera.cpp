#include "SemanticSegmentationCamera.h"

USemanticSegmentationCamera::USemanticSegmentationCamera() {
	SensorName = "SemanticSegmentationCamera";

	AddPostProcessingMaterial(TEXT(
		"/Script/Engine.Material'/Game/PostProcessingMaterials/"
		"PhysicLensDistortion.PhysicLensDistortion'"));
	AddPostProcessingMaterial(TEXT(
		"/Script/Engine.Material'/Game/PostProcessingMaterials/"
		"GTMaterial.GTMaterial'"));
}

void USemanticSegmentationCamera::ParseSensorParms(FString ParmsJson) {
	Super::ParseSensorParms(ParmsJson);
}

void USemanticSegmentationCamera::InitializeSensor() {
	Super::InitializeSensor();
}

void USemanticSegmentationCamera::TickSensorComponent(
	float						 DeltaTime,
	ELevelTick					 TickType,
	FActorComponentTickFunction* ThisTickFunction) {

	TickCounter++;
	if (TickCounter >= TicksPerCapture) {
		RenderRequest.RetrievePixels(Buffer, TargetTexture, true);
		TickCounter -= TicksPerCapture;
	}
}
