// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
#pragma once

#include "SensorShader.generated.h"

/// A shader in UShaderBasedSensor.
USTRUCT(BlueprintType)
struct HOLODECK_API FSensorShader {
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	UMaterialInstanceDynamic* PostProcessMaterial = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Weight = 1.0f;
};