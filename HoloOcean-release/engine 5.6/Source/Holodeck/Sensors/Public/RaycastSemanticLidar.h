// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
#pragma once

#include <vector>

#include "Holodeck.h"
#include "HolodeckSensor.h"
#include "LidarData.h"
#include "LidarDescription.h"
#include "RandomEngine.h"
#include "SemanticLidarData.h"
#include "Tagger.h"
#include "RaycastLidar.h"
#include "GameplayTagsManager.h"
#include "Landscape.h"
#include "LandscapeComponent.h"
#include "LandscapeHeightfieldCollisionComponent.h"

#include "RaycastSemanticLidar.generated.h"
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))

class HOLODECK_API URaycastSemanticLidar : public URaycastLidar {
	GENERATED_BODY()

	using FSemanticLidarData = holoocean::data::SemanticLidarData;
	using FSemanticDetection = holoocean::data::SemanticLidarDetection;

public:
	/*
	 * Default Constructor
	 */
	URaycastSemanticLidar();

	/**
	 * InitializeSensor
	 * Sets up the class
	 */
	void InitializeSensor() override;

protected:
	/// This method uses all the saved FHitResults, computes the RawDetections and then
	/// sends it to the LidarData structure.
	void ComputeAndSaveDetections(const FTransform& SensorTransform) override;

	/// Start the tick function for the sensor component.
	void TickSensorComponent(
		float						 DeltaTime,
		ELevelTick					 TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

	std::vector<FSemanticDetection> PrevDetections;
	std::vector<FVector>			WorldPoints;

private:
	/// Compute all raw detection information
	FSemanticDetection ComputeRawDetection(
		const FHitResult& HitInfo,
		const FTransform& SensorTransf) const;

	/// Get the color from the semantic tag.
	FColor GetColorFromSemanticTag(int32 Tag);

	/// Get the tag type for landscape materials.
	int32 GetTagFromSurfaceType(EPhysicalSurface SurfaceType) const;

	/// After initialization, Parent contains a pointer to whatever the sensor is
	/// attached to.
	TUniquePtr<AActor> Parent;
	float*			   SemanticLidarBuffer;
	FSemanticLidarData Data;
};
