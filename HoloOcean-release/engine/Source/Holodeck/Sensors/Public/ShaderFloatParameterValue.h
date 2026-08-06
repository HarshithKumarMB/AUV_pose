// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
#pragma once

#include "ShaderFloatParameterValue.generated.h"

/// A shader parameter value to change when the material
/// instance is available.

USTRUCT(BlueprintType)
struct HOLODECK_API FShaderFloatParameterValue {
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int ShaderIndex = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FName ParameterName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Value = 0.0f;
};