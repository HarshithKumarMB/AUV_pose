#include "RaycastSemanticLidar.h"

#include <cmath>

URaycastSemanticLidar::URaycastSemanticLidar() {
	SensorName = "RaycastSemanticLidar";
	Parent = nullptr;
}

void URaycastSemanticLidar::InitializeSensor() {

	Super::ShouldInitialize = false;
	Super::InitializeSensor();

	SemanticLidarBuffer = static_cast<float*>(Buffer);
	uint32_t NumPointComponents =
		7u; // 7 components: x, y, z, intensity, ring, object_idx, object_tag
	Data = FSemanticLidarData(
		SemanticLidarBuffer, Description.Channels, NumPointComponents);

	Parent = TUniquePtr<AActor>(GetAttachmentRootActor());

	if (RandomEngine && Description.RandomSeed > 0) {
		RandomEngine->Seed(Description.RandomSeed);
	}
}

void URaycastSemanticLidar::TickSensorComponent(
	float						 DeltaTime,
	ELevelTick					 TickType,
	FActorComponentTickFunction* ThisTickFunction) {
	this->DeltaTime = DeltaTime;
	TickCounter++;
	if (TickCounter == TicksPerCapture) {
		SimulateLidar(DeltaTime);
		TickCounter = 0;
	} else {

		const uint32 ChannelCount = Description.Channels;
		const uint32 PointsToScanWithOneLaser = FMath::RoundHalfFromZero(
			Description.PointsPerSecond * (DeltaTime * TicksPerCapture)
			/ static_cast<float>(ChannelCount));
		ResetRecordedHits(ChannelCount, PointsToScanWithOneLaser);
		PreprocessRays(ChannelCount, PointsToScanWithOneLaser);
		FTransform ActorTransf = this->GetComponentTransform();
		ComputeAndSaveDetections(ActorTransf);
	}
}

void URaycastSemanticLidar::ComputeAndSaveDetections(
	const FTransform& SensorTransform) {
	for (auto idxChannel = 0u; idxChannel < Description.Channels; ++idxChannel) {
		PointsPerChannel[idxChannel] = RecordedHits[idxChannel].size();
	}

	Data.ResetMemory(PointsPerChannel);

	int PointCount = 0;
	for (auto idxChannel = 0u; idxChannel < Description.Channels; ++idxChannel) {
		for (auto& hit : RecordedHits[idxChannel]) {
			FSemanticDetection Detection = ComputeRawDetection(hit, SensorTransform);
			Detection.ring = idxChannel; // Set the ring to the channel index

			if (PostprocessDetection(Detection)) {
				Data.WritePointSync(Detection);
				PointCount++;

				if (Description.ShowDebugPoints) {
					const FVector WorldPoint =
						SensorTransform.TransformPosition(FVector(Detection.point));
					DrawDebugPoint(
						GetWorld(),
						WorldPoint,
						4.0f,
						GetColorFromSemanticTag(
							Detection.object_tag), // now colored by tag
						// FColor::White, // Default to white for debug points
						false,
						TicksPerCapture * DeltaTime);
				}
			} else {
				PointsPerChannel[idxChannel]--;
			}
		}
	}
	SemanticLidarBuffer[0] = PointCount;
}

URaycastSemanticLidar::FSemanticDetection URaycastSemanticLidar::ComputeRawDetection(
	const FHitResult& HitInfo,
	const FTransform& SensorTransf) const {
	FSemanticDetection Detection;

	// Convert the hit point from world coordinates to sensor local coordinates
	const FVector HitPoint = HitInfo.ImpactPoint;
	Detection.point =
		static_cast<FVector3d>(SensorTransf.Inverse().TransformPosition(HitPoint));

	// Compute intensity
	const float Distance = Detection.point.Length();
	const float AttenAtm = Description.AtmospAttenRate;
	const float AbsAtm = exp(-AttenAtm * Distance);
	const float IntRec = AbsAtm;
	Detection.intensity = IntRec;

	// Get the instance ID from the hit component
	Detection.object_idx =
		HitInfo.GetActor()
			->GetUniqueID(); // Get the unique ID of the actor that was hit
	// Get the semantic label from the hit component
	Detection.object_tag =
		static_cast<uint32_t>(HitInfo.Component->CustomDepthStencilValue);

	// Need to check if hit was a landscape (different then a static mesh, so need extra
	// steps to get the landscape component)
	ULandscapeHeightfieldCollisionComponent* CollisionComponent =
		Cast<ULandscapeHeightfieldCollisionComponent>(HitInfo.Component);
	if (CollisionComponent != nullptr) {
		ULandscapeComponent* RenderComponent = CollisionComponent->GetRenderComponent();
		if (RenderComponent != nullptr) {
			Detection.object_tag =
				static_cast<uint32_t>(RenderComponent->CustomDepthStencilValue);
		}

		// Some landscapes have physical materials to differentiate between ground types
		// If so, set the object_tag to the corresponding surface type found in the
		// Project Settings
		UPhysicalMaterial* HitPhysicalMaterial = HitInfo.PhysMaterial.Get();
		if (HitPhysicalMaterial) {
			EPhysicalSurface SurfaceType = HitPhysicalMaterial->SurfaceType;
			Detection.object_tag = GetTagFromSurfaceType(SurfaceType);
		}
	}

	return Detection;
}

int32 URaycastSemanticLidar::GetTagFromSurfaceType(EPhysicalSurface SurfaceType) const {
	switch (SurfaceType) {
		case EPhysicalSurface::SurfaceType1:
			return static_cast<uint32_t>(32);
		case EPhysicalSurface::SurfaceType2:
			return static_cast<uint32_t>(33);
		case EPhysicalSurface::SurfaceType3:
			return static_cast<uint32_t>(34);
		case EPhysicalSurface::SurfaceType4:
			return static_cast<uint32_t>(35);
		default:
			return static_cast<uint32_t>(0);
	}
}

FColor URaycastSemanticLidar::GetColorFromSemanticTag(int32 Tag) {
	switch (Tag) {
		case static_cast<int32>(holoocean::ObjectLabel::Asphalt):
			return FColor(79, 82, 77);
		case static_cast<int32>(holoocean::ObjectLabel::Bench):
			return FColor(212, 191, 125);
		case static_cast<int32>(holoocean::ObjectLabel::BikeRack):
			return FColor(209, 208, 203);
		case static_cast<int32>(holoocean::ObjectLabel::Building):
			return FColor(31, 240, 205);
		case static_cast<int32>(holoocean::ObjectLabel::Bus):
			return FColor(240, 220, 43);
		case static_cast<int32>(holoocean::ObjectLabel::Bush):
			return FColor(177, 247, 124);
		case static_cast<int32>(holoocean::ObjectLabel::Car):
			return FColor(224, 109, 237);
		case static_cast<int32>(holoocean::ObjectLabel::Ceiling):
			return FColor(161, 247, 233);
		case static_cast<int32>(holoocean::ObjectLabel::Chair):
			return FColor(65, 13, 255);
		case static_cast<int32>(holoocean::ObjectLabel::Cone):
			return FColor(255, 169, 10);
		case static_cast<int32>(holoocean::ObjectLabel::Crate):
			return FColor(140, 86, 0);
		case static_cast<int32>(holoocean::ObjectLabel::Desk):
			return FColor(242, 65, 224);
		case static_cast<int32>(holoocean::ObjectLabel::Dumpster):
			return FColor(0, 138, 14);
		case static_cast<int32>(holoocean::ObjectLabel::FireHydrant):
			return FColor(255, 0, 0);
		case static_cast<int32>(holoocean::ObjectLabel::Floor):
			return FColor(2, 125, 104);
		case static_cast<int32>(holoocean::ObjectLabel::GarbageCan):
			return FColor(0, 184, 92);
		case static_cast<int32>(holoocean::ObjectLabel::Grass):
			return FColor(91, 235, 52);
		case static_cast<int32>(holoocean::ObjectLabel::Pallet):
			return FColor(143, 129, 6);
		case static_cast<int32>(holoocean::ObjectLabel::ParkingGate):
			return FColor(222, 164, 177);
		case static_cast<int32>(holoocean::ObjectLabel::PatioUmbrella):
			return FColor(195, 0, 255);
		case static_cast<int32>(holoocean::ObjectLabel::Railing):
			return FColor(242, 97, 130);
		case static_cast<int32>(holoocean::ObjectLabel::SemiTruck):
			return FColor(124, 6, 138);
		case static_cast<int32>(holoocean::ObjectLabel::Sidewalk):
			return FColor(232, 232, 232);
		case static_cast<int32>(holoocean::ObjectLabel::SpeedLimitSign):
			return FColor(222, 218, 245);
		case static_cast<int32>(holoocean::ObjectLabel::StopSign):
			return FColor(250, 37, 62);
		case static_cast<int32>(holoocean::ObjectLabel::StreetLamps):
			return FColor(255, 149, 10);
		case static_cast<int32>(holoocean::ObjectLabel::Table):
			return FColor(186, 17, 169);
		case static_cast<int32>(holoocean::ObjectLabel::Tree):
			return FColor(69, 99, 46);
		case static_cast<int32>(holoocean::ObjectLabel::Unlabeled):
			return FColor(255, 255, 255);
		case static_cast<int32>(holoocean::ObjectLabel::Wall):
			return FColor(126, 60, 250);
		case static_cast<int32>(holoocean::ObjectLabel::Trash):
			return FColor(107, 156, 137);
		case static_cast<int32>(holoocean::ObjectLabel::GroundGrass):
			return FColor(91, 235, 52);
		case static_cast<int32>(holoocean::ObjectLabel::GroundRock):
			return FColor(196, 101, 6);
		case static_cast<int32>(holoocean::ObjectLabel::Ground):
			return FColor(194, 164, 159);
		case static_cast<int32>(holoocean::ObjectLabel::GroundPath):
			return FColor(242, 219, 196);
		case static_cast<int32>(holoocean::ObjectLabel::Boat):
			return FColor(128, 64, 128);
		case static_cast<int32>(holoocean::ObjectLabel::Yacht):
			return FColor(70, 70, 70);
		case static_cast<int32>(holoocean::ObjectLabel::ContainerBoat):
			return FColor(102, 102, 156);
		// case static_cast<int32>(holoocean::ObjectLabel::DamEnvironment): 	return
		// FColor(190, 153, 153);
		case static_cast<int32>(holoocean::ObjectLabel::Landscape):
			return FColor(153, 153, 153);
		default:
			return FColor::White;
	}
}