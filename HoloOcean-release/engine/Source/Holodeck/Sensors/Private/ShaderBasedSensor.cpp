#include "ShaderBasedSensor.h"

bool UShaderBasedSensor::AddPostProcessingMaterial(const FString& Path) {
	ConstructorHelpers::FObjectFinder<UMaterial> Loader(*Path);

	if (Loader.Succeeded()) {
		MaterialsFound.Add(Loader.Object);
	}

	return Loader.Succeeded();
}

void UShaderBasedSensor::SetUpSceneCaptureComponent() {
	for (const auto& MaterialFound : MaterialsFound) {
		// Create a dynamic instance of the Material (Shader)
		AddShader({ UMaterialInstanceDynamic::Create(MaterialFound, this), 1.0f });
	}

	for (const auto& Shader : Shaders) {
		// Attach the instance into the blendables
		SceneCapture->PostProcessSettings.AddBlendable(
			Shader.PostProcessMaterial, Shader.Weight);
	}

	// Set the value for each Float parameter in the shader
	for (const auto& ParameterValue : FloatShaderParams) {
		Shaders[ParameterValue.ShaderIndex]
			.PostProcessMaterial->SetScalarParameterValue(
				ParameterValue.ParameterName, ParameterValue.Value);
	}
}

void UShaderBasedSensor::SetFloatShaderParameter(
	uint8_t		 ShaderIndex,
	const FName& ParameterName,
	float		 Value) {
	FloatShaderParams.Add({ ShaderIndex, ParameterName, Value });
}

void UShaderBasedSensor::InitializeSensor() {
	Super::InitializeSensor();
	SetUpSceneCaptureComponent();
}

void UShaderBasedSensor::ParseSensorParms(FString ParmsJson) {
	Super::ParseSensorParms(ParmsJson);

	TSharedPtr<FJsonObject>		   JsonParsed;
	TSharedRef<TJsonReader<TCHAR>> JsonReader =
		TJsonReaderFactory<TCHAR>::Create(ParmsJson);
	if (FJsonSerializer::Deserialize(JsonReader, JsonParsed)) {

		if (JsonParsed->HasTypedField<EJson::Number>("TicksPerCapture")) {
			TicksPerCapture = JsonParsed->GetIntegerField("TicksPerCapture");
		}
	} else {
		UE_LOG(
			LogHolodeck,
			Fatal,
			TEXT("UShaderBasedSensor::ParseSensorParms:: Unable to parse json."));
	}
}
