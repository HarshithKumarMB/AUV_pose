
#pragma once
#include "CameraSensor.h"
#include "SensorShader.h"
#include "ShaderFloatParameterValue.h"

#include "ShaderBasedSensor.generated.h"

UCLASS(Abstract)
class HOLODECK_API UShaderBasedSensor : public UCameraSensor {
	GENERATED_BODY()

public:
	UShaderBasedSensor() {
		SensorName = "ShaderBasedSensor (DO NOT INSTANTIATE BY ITSELF!)";
	};

	virtual void InitializeSensor() override;

	/**
	 * Allows parameters to be set dynamically
	 **/
	virtual void ParseSensorParms(FString ParmsJson) override;

	/// Load the UMaterialInstanceDynamic at the given @a Path and
	/// append it to the list of shaders with @a Weight.
	///
	/// @return Whether it succeeded.
	UFUNCTION(BlueprintCallable)
	// bool LoadPostProcessingMaterial(const FString &Path, float Weight = 1.0f);
	bool AddPostProcessingMaterial(const FString& Path);

	// Add a post-processing shader.
	UFUNCTION(BlueprintCallable)
	void AddShader(const FSensorShader& Shader) { Shaders.Add(Shader); }

	void SetFloatShaderParameter(
		uint8_t		 ShaderIndex,
		const FName& ParameterName,
		float		 Value);

protected:
	void SetUpSceneCaptureComponent();

	virtual void TickSensorComponent(
		float						 DeltaTime,
		ELevelTick					 TickType,
		FActorComponentTickFunction* ThisTickFunction) override {};

private:
	int TickCounter = 0;

	UPROPERTY()
	TArray<UMaterial*> MaterialsFound;

	UPROPERTY()
	TArray<FSensorShader> Shaders;

	UPROPERTY()
	TArray<FShaderFloatParameterValue> FloatShaderParams;
};
