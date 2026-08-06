#include "RaycastLidar.h"

#include <Engine/CollisionProfile.h>
#include <cmath>

#include "RandomEngine.h"

URaycastLidar::URaycastLidar() {
	SensorName = "RaycastLidar";
	Parent = nullptr;
	DropOffAlpha = 0.0f;
	DropOffBeta = 0.0f;
	DropOffGenActive = false;

	RandomEngine = CreateDefaultSubobject<URandomEngine>(TEXT("RandomEngine"));
}

void URaycastLidar::ParseSensorParms(FString ParmsJson) {
	// Parse all the configuration options for the lidar (num lasers, etc.)
	Description = FLidarDescription();

	TSharedPtr<FJsonObject>		   JsonParsed;
	TSharedRef<TJsonReader<TCHAR>> JsonReader =
		TJsonReaderFactory<TCHAR>::Create(ParmsJson);

	if (FJsonSerializer::Deserialize(JsonReader, JsonParsed)) {
		JsonParsed->TryGetNumberField("Channels", Description.Channels);
		JsonParsed->TryGetNumberField("Range", Description.Range);
		JsonParsed->TryGetNumberField("PointsPerSecond", Description.PointsPerSecond);
		JsonParsed->TryGetNumberField(
			"RotationFrequency", Description.RotationFrequency);
		JsonParsed->TryGetNumberField("UpperFovLimit", Description.UpperFovLimit);
		JsonParsed->TryGetNumberField("LowerFovLimit", Description.LowerFovLimit);
		JsonParsed->TryGetNumberField("HorizontalFov", Description.HorizontalFov);
		JsonParsed->TryGetNumberField("AtmospAttenRate", Description.AtmospAttenRate);
		JsonParsed->TryGetNumberField("RandomSeed", Description.RandomSeed);
		JsonParsed->TryGetNumberField("DropOffGenRate", Description.DropOffGenRate);
		JsonParsed->TryGetNumberField(
			"DropOffIntensityLimit", Description.DropOffIntensityLimit);
		JsonParsed->TryGetNumberField(
			"DropOffAtZeroIntensity", Description.DropOffAtZeroIntensity);
		JsonParsed->TryGetBoolField("ShowDebugPoints", Description.ShowDebugPoints);
		JsonParsed->TryGetNumberField("NoiseStdDev", Description.NoiseStdDev);

		if (JsonParsed->HasTypedField<EJson::Number>("TicksPerCapture")) {
			TicksPerCapture = JsonParsed->GetIntegerField("TicksPerCapture");
		}

		Description.Range *=
			100; // We need to convert it to cm since that is what HoloOcean uses
		Description.NoiseStdDev *= 100;		// Convert noise std dev to cm as well
		Description.AtmospAttenRate /= 100; // Convert atmosp atten rate to cm^-1
	}

	Set(Description);
}

void URaycastLidar::InitializeSensor() {
	Super::InitializeSensor();

	Parent = TUniquePtr<AActor>(this->GetAttachmentRootActor());

	if (ShouldInitialize) {
		ShouldInitialize = false;
		LidarBuffer = static_cast<float*>(Super::Buffer);

		uint32_t NumPointComponents = 5u; // 5 components: x, y, z, intensity, ring
		Data = FLidarData(LidarBuffer, Description.Channels, NumPointComponents);
	}

	if (RandomEngine && Description.RandomSeed > 0) {
		RandomEngine->Seed(Description.RandomSeed);
	}
}

void URaycastLidar::TickSensorComponent(
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

void URaycastLidar::Set(const FLidarDescription& LidarDescription) {
	Description = LidarDescription;

	CreateLasers();
	PointsPerChannel.resize(Description.Channels);

	// Compute drop off model parameters
	DropOffBeta = Description.DropOffAtZeroIntensity;
	DropOffAlpha =
		(1.0f - Description.DropOffAtZeroIntensity) / Description.DropOffIntensityLimit;
	DropOffGenActive =
		Description.DropOffGenRate > std::numeric_limits<float>::epsilon();
}

/**
 * @brief Initializes the vertical angles for each lidar laser channel.
 *
 * This function calculates and stores the vertical angles for all laser channels based
 * on the lidar's field of view (FOV) configuration. The angles are distributed evenly
 * between the UpperFovLimit and LowerFovLimit specified in the lidar description. The
 * resulting angles are stored in the LaserAngles array, which is used for simulating
 * the lidar rays.
 *
 * @note The number of lasers (channels) must be greater than zero.
 * @see FLidarDescription
 */
void URaycastLidar::CreateLasers() {
	const auto NumberOfLasers = Description.Channels;
	check(NumberOfLasers > 0u);

	const float DeltaAngle = NumberOfLasers == 1u
		? 0.f
		: (Description.UpperFovLimit - Description.LowerFovLimit)
			/ static_cast<float>(NumberOfLasers - 1);

	LaserAngles.Empty(NumberOfLasers);

	for (auto i = 0u; i < NumberOfLasers; ++i) {
		const float VerticalAngle =
			Description.UpperFovLimit - static_cast<float>(i) * DeltaAngle;
		LaserAngles.Emplace(VerticalAngle);
	}
}

void URaycastLidar::SimulateLidar(const float DeltaTime) {
	const uint32 ChannelCount = Description.Channels;
	const uint32 PointsToScanWithOneLaser = FMath::RoundHalfFromZero(
		Description.PointsPerSecond * (DeltaTime * TicksPerCapture)
		/ static_cast<float>(ChannelCount));

	if (PointsToScanWithOneLaser <= 0) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"%s: no points request this frame.. Try increasing the number of points per second."),
			*GetName());
		return;
	}

	check(ChannelCount == LaserAngles.Num());

	const float CurrentHorizontalAngle = Data.GetHorizontalAngle();
	const float AngleDistanceOfTick = Description.RotationFrequency
		* Description.HorizontalFov
		* (DeltaTime * TicksPerCapture); // Calculate the "slice" of the Horizontal FOV
										 // that we are scanning this tick
	const float AngleDistanceOfLaserMeasure = AngleDistanceOfTick
		/ PointsToScanWithOneLaser; // Calculate the horizontal angle between each laser
									// measure

	ResetRecordedHits(ChannelCount, PointsToScanWithOneLaser);
	PreprocessRays(ChannelCount, PointsToScanWithOneLaser);

	{
		SCOPE_CYCLE_COUNTER(STAT_ParallelForHoloocean)
		ParallelFor(ChannelCount, [&](int32 idxChannel) {
			TRACE_CPUPROFILER_EVENT_SCOPE(STAT_ParallelFor);

			// TODO: Parent was "this" before, but since the sensors are not actors, it
			// couldn't be. This might cause issues though if it gets garbage collected
			FCollisionQueryParams TraceParams =
				FCollisionQueryParams(FName(TEXT("Laser_Trace")), true, Parent.Get());
			TraceParams.bTraceComplex = true;
			TraceParams.bReturnPhysicalMaterial = true; // false;

			for (auto idxPtsOneLaser = 0u; idxPtsOneLaser < PointsToScanWithOneLaser;
				 idxPtsOneLaser++) {
				FHitResult	HitResult;
				const float VertAngle = LaserAngles[idxChannel];
				const float HorizAngle =
					std::fmod(
						CurrentHorizontalAngle
							+ AngleDistanceOfLaserMeasure * idxPtsOneLaser,
						Description.HorizontalFov)
					- Description.HorizontalFov / 2;
				const bool PreprocessResult =
					RayPreprocessCondition[idxChannel][idxPtsOneLaser];

				if (PreprocessResult
					&& ShootLaser(VertAngle, HorizAngle, HitResult, TraceParams)) {
					WritePointAsync(idxChannel, HitResult);
				}
			};
		});
	}

	FTransform ActorTransf = this->GetComponentTransform();
	ComputeAndSaveDetections(ActorTransf);

	// Calculate the new horizontal angle for the next tick in degrees
	const float HorizontalAngle = std::fmod(
		CurrentHorizontalAngle + AngleDistanceOfTick, Description.HorizontalFov);
	Data.SetHorizontalAngle(HorizontalAngle);
}

void URaycastLidar::ResetRecordedHits(uint32_t Channels, uint32_t MaxPointsPerChannel) {
	RecordedHits.resize(Channels);

	for (auto& hits : RecordedHits) {
		hits.clear();
		hits.reserve(MaxPointsPerChannel);
	}
}

void URaycastLidar::PreprocessRays(uint32_t Channels, uint32_t MaxPointsPerChannel) {
	RayPreprocessCondition.resize(Channels);

	for (uint32_t ch = 0; ch < Channels; ++ch) {
		RayPreprocessCondition[ch].resize(MaxPointsPerChannel);
		for (uint32_t p = 0; p < MaxPointsPerChannel; ++p) {
			RayPreprocessCondition[ch][p] =
				!(DropOffGenActive
				  && RandomEngine->GetUniformFloat() < Description.DropOffGenRate);
		}
	}
}

bool URaycastLidar::PostprocessDetection(FDetection& Detection) const {
	if (Description.NoiseStdDev > std::numeric_limits<float>::epsilon()) {
		// Generate noise in the forward direction of the lidar
		const auto ForwardVector = Detection.point.GetSafeNormal();
		const auto Noise = ForwardVector
			* RandomEngine->GetNormalDistribution(0.0f, Description.NoiseStdDev);
		Detection.point += Noise;
	}

	// If the drop off is not active, we keep all the points
	float Intensity = Detection.intensity;
	if (Intensity > Description.DropOffIntensityLimit) {
		return true;
	}

	// Else we apply the drop off linear model
	return RandomEngine->GetUniformFloat() < DropOffAlpha * Intensity + DropOffBeta;
}

void URaycastLidar::WritePointAsync(uint32_t Channel, FHitResult& Detection) {
	RecordedHits[Channel].emplace_back(Detection);
}

void URaycastLidar::ComputeAndSaveDetections(const FTransform& SensorTransform) {
	for (auto idxChannel = 0u; idxChannel < Description.Channels; ++idxChannel) {
		PointsPerChannel[idxChannel] = RecordedHits[idxChannel].size();
	}
	Data.ResetMemory(PointsPerChannel);
	int PointCount = 0;
	for (auto idxChannel = 0u; idxChannel < Description.Channels; ++idxChannel) {
		for (auto& hit : RecordedHits[idxChannel]) {
			FDetection Detection = ComputeRawDetection(hit, SensorTransform);
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
						FColor::White,
						false,
						TicksPerCapture * DeltaTime);
				}
			} else {
				PointsPerChannel[idxChannel]--;
			}
		}
	}
	LidarBuffer[0] = PointCount;
}

URaycastLidar::FDetection URaycastLidar::ComputeRawDetection(
	const FHitResult& HitInfo,
	const FTransform& SensorTransf) const {
	FDetection	  Detection;
	const FVector HitPoint = HitInfo.ImpactPoint;
	Detection.point = SensorTransf.Inverse().TransformPosition(HitPoint);

	// Compute intensity
	const float Distance = Detection.point.Length();
	const float AttenAtm = Description.AtmospAttenRate;
	const float AbsAtm = exp(-AttenAtm * Distance);
	const float IntRec = AbsAtm;
	Detection.intensity = IntRec;

	return Detection;
}

bool URaycastLidar::ShootLaser(
	const float			   VerticalAngle,
	float				   HorizontalAngle,
	FHitResult&			   HitResult,
	FCollisionQueryParams& TraceParams) const {
	FHitResult HitInfo(ForceInit);
	FTransform ActorTransf = this->GetComponentTransform();
	FVector	   LidarBodyLoc = ActorTransf.GetLocation();

	FRotator LidarBodyRot = ActorTransf.Rotator();

	FRotator LaserRot(
		VerticalAngle, HorizontalAngle, 0); // float InPitch, float InYaw, float InRoll
	FRotator ResultRot = UKismetMathLibrary::ComposeRotators(LaserRot, LidarBodyRot);

	const auto Range = Description.Range;
	FVector	   EndTrace =
		Range * UKismetMathLibrary::GetForwardVector(ResultRot) + LidarBodyLoc;

	GetWorld()->LineTraceSingleByChannel(
		HitInfo,
		LidarBodyLoc,
		EndTrace,
		ECC_Visibility,
		TraceParams,
		FCollisionResponseParams::DefaultResponseParam);

	if (HitInfo.bBlockingHit) {
		HitResult = HitInfo;
		return true;
	}

	return false;
}
