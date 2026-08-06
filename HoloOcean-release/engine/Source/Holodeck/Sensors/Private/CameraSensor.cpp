#include "CameraSensor.h"

#include "Engine.h"

#include "Engine/SceneCapture2D.h"

#include "RHICommandList.h"

#include "Modules/ModuleManager.h"

UCameraSensor::UCameraSensor() {
	SensorName = "CameraSensor";
	Parent = nullptr;
}

void UCameraSensor::ParseSensorParms(FString ParmsJson) {
	Super::ParseSensorParms(ParmsJson);

	SceneCapture = NewObject<USceneCaptureComponent2D>(this, "SceneCap");
	SceneCapture->RegisterComponent();
	SceneCapture->AttachToComponent(
		this, FAttachmentTransformRules::KeepRelativeTransform);

	TSharedPtr<FJsonObject>		   JsonParsed;
	TSharedRef<TJsonReader<TCHAR>> JsonReader =
		TJsonReaderFactory<TCHAR>::Create(ParmsJson);
	if (FJsonSerializer::Deserialize(JsonReader, JsonParsed)) {

		JsonParsed->TryGetNumberField("TicksPerCapture", TicksPerCapture);

		if (JsonParsed->TryGetNumberField("FovAngle", FovAngle)) {
			SetFOVAngle(FovAngle);
		}

		if (JsonParsed->TryGetNumberField("TargetGamma", TargetGamma)) {
			SetTargetGamma(TargetGamma);
		}

		if (FString exposureMethod;
			JsonParsed->TryGetStringField("ExposureMethod", exposureMethod)) {
			if (exposureMethod == "AEM_Histogram") {
				SetExposureMethod(AEM_Histogram);
			} else if (exposureMethod == "AEM_Basic") {
				SetExposureMethod(AEM_Basic);
			} else if (exposureMethod == "AEM_Manual") {
				SetExposureMethod(AEM_Manual);
			} else if (exposureMethod == "AEM_MAX") {
				SetExposureMethod(AEM_MAX);
			} else {
				UE_LOG(
					LogHolodeck,
					Error,
					TEXT("%s is NOT a valid Exposure Method"),
					*exposureMethod);
			}
		}
		if (JsonParsed->TryGetNumberField(
				"ExposureCompensation", ExposureCompensation)) {
			SetExposureCompensation(ExposureCompensation);
		} else {
			// Higher = brighter captured image. Lower = darker
			// This is a magic number that has been fine-tuned to the default worlds.
			// Do not edit without thorough testing.
			SetExposureCompensation(4);
		}

		if (JsonParsed->TryGetNumberField("ShutterSpeed", ShutterSpeed)) {
			SetShutterSpeed(ShutterSpeed);
		}

		if (JsonParsed->TryGetNumberField("ISO", Iso)) {
			SetISO(Iso);
		}

		if (JsonParsed->TryGetNumberField("Aperture", Aperture)) {
			SetAperture(Aperture);
		}

		if (JsonParsed->TryGetNumberField("FocalDistance", FocalDistance)) {
			SetFocalDistance(FocalDistance);
		}

		if (JsonParsed->TryGetNumberField("DepthBlurAmount", DepthBlurAmount)) {
			SetDepthBlurAmount(DepthBlurAmount);
		}

		if (JsonParsed->TryGetNumberField("DepthBlurRadius", DepthBlurRadius)) {
			SetDepthBlurRadius(DepthBlurRadius);
		}

		if (JsonParsed->TryGetNumberField("BladeCount", BladeCount)) {
			SetBladeCount(BladeCount);
		}

		if (JsonParsed->TryGetNumberField(
				"DepthOfFieldMinFstop", DepthOfFieldMinFstop)) {
			SetDepthOfFieldMinFstop(DepthOfFieldMinFstop);
		}

		if (JsonParsed->TryGetNumberField("FilmSlope", FilmSlope)) {
			SetFilmSlope(FilmSlope);
		}

		if (JsonParsed->TryGetNumberField("FilmToe", FilmToe)) {
			SetFilmToe(FilmToe);
		}

		if (JsonParsed->TryGetNumberField("FilmShoulder", FilmShoulder)) {
			SetFilmShoulder(FilmShoulder);
		}

		if (JsonParsed->TryGetNumberField("FilmBlackClip", FilmBlackClip)) {
			SetFilmBlackClip(FilmBlackClip);
		}

		if (JsonParsed->TryGetNumberField("FilmWhiteClip", FilmWhiteClip)) {
			SetFilmWhiteClip(FilmWhiteClip);
		}

		if (JsonParsed->TryGetNumberField(
				"ExposureMinBrightness", ExposureMinBrightness)) {
			SetExposureMinBrightness(ExposureMinBrightness);
		}

		if (JsonParsed->TryGetNumberField(
				"ExposureMaxBrightness", ExposureMinBrightness)) {
			SetExposureMinBrightness(ExposureMaxBrightness);
		}

		if (JsonParsed->TryGetNumberField("ExposureSpeedDown", ExposureSpeedDown)) {
			SetExposureSpeedDown(ExposureSpeedDown);
		}

		if (JsonParsed->TryGetNumberField("ExposureSpeedUp", ExposureSpeedUp)) {
			SetExposureSpeedUp(ExposureSpeedUp);
		}

		if (JsonParsed->TryGetNumberField("MotionBlurIntensity", MotionBlurIntensity)) {
			SetMotionBlurIntensity(MotionBlurIntensity);
		}

		if (JsonParsed->TryGetNumberField(
				"MotionBlurMaxDistortion", MotionBlurMaxDistortion)) {
			SetMotionBlurMaxDistortion(MotionBlurMaxDistortion);
		}

		if (JsonParsed->TryGetNumberField(
				"MotionBlurMinObjectScreenSize", MotionBlurMinObjectScreenSize)) {
			SetMotionBlurMinObjectScreenSize(MotionBlurMinObjectScreenSize);
		}

		if (JsonParsed->TryGetNumberField("LensFlareIntensity", LensFlareIntensity)) {
			SetLensFlareIntensity(LensFlareIntensity);
		}

		if (JsonParsed->TryGetNumberField("BloomIntensity", BloomIntensity)) {
			SetBloomIntensity(BloomIntensity);
		}

		if (JsonParsed->TryGetNumberField("WhiteTemp", WhiteTemp)) {
			SetWhiteTemp(WhiteTemp);
		}

		if (JsonParsed->TryGetNumberField("WhiteTint", WhiteTint)) {
			SetWhiteTemp(WhiteTint);
		}

		if (JsonParsed->TryGetNumberField("ChromAberrIntensity", ChromAberrIntensity)) {
			SetChromAberrIntensity(ChromAberrIntensity);
		}

		if (JsonParsed->TryGetNumberField("ChromAberrOffset", ChromAberrOffset)) {
			SetChromAberrOffset(ChromAberrOffset);
		}

		if (JsonParsed->TryGetNumberField("MaxViewDistanceOverride", ViewDistance)) {
			SetViewDistance(ViewDistance);
		} else {
			/*ViewDistance = 10000;
	  SetViewDistance(ViewDistance);*/
		}
	} else {
		UE_LOG(
			LogHolodeck,
			Fatal,
			TEXT("UCameraSensor::ParseSensorParms:: Unable to parse json."));
	}
}

void UCameraSensor::InitializeSensor() {
	Super::InitializeSensor();

	Parent = this->GetAttachmentRootActor();

	// Set up everything for the SceneCaptureComponent2D
	SceneCapture->CaptureSource =
		SCS_FinalColorLDR; // Pick what type of output you want to be sent to the
						   // texture target.
}

void UCameraSensor::TickSensorComponent(
	float						 DeltaTime,
	ELevelTick					 TickType,
	FActorComponentTickFunction* ThisTickFunction) {

	TickCounter++;
	if (TickCounter >= TicksPerCapture) {
		RenderRequest.RetrievePixels(Buffer, TargetTexture);
		TickCounter -= TicksPerCapture;
	}
}

void UCameraSensor::SetFOVAngle(float FOVAngle) {
	check(this->SceneCapture != nullptr);
	this->SceneCapture->FOVAngle = FOVAngle;
}

float UCameraSensor::GetFOVAngle() const {
	check(this->SceneCapture != nullptr);
	return this->SceneCapture->FOVAngle;
}

void UCameraSensor::SetViewDistance(float viewDistance) {
	check(this->SceneCapture != nullptr);
	this->SceneCapture->MaxViewDistanceOverride = viewDistance;
}

float UCameraSensor::GetViewDistance() const {
	check(this->SceneCapture != nullptr);
	return this->SceneCapture->MaxViewDistanceOverride;
}

void UCameraSensor::SetExposureMethod(EAutoExposureMethod Method) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.AutoExposureMethod = Method;
}

EAutoExposureMethod UCameraSensor::GetExposureMethod() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.AutoExposureMethod;
}

void UCameraSensor::SetExposureCompensation(float Compensation) {
	check(SceneCapture != nullptr);

#if PLATFORM_LINUX
	// Looks like Windows and Linux have different outputs with the
	// same exposure compensation, this fixes it.
	SceneCapture->PostProcessSettings.AutoExposureBias = Compensation + 0.75f;
#else
	SceneCapture->PostProcessSettings.AutoExposureBias = Compensation;
#endif
}

float UCameraSensor::GetExposureCompensation() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.AutoExposureBias;
}

void UCameraSensor::SetShutterSpeed(float Speed) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.CameraShutterSpeed = Speed;
}

float UCameraSensor::GetShutterSpeed() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.CameraShutterSpeed;
}

void UCameraSensor::SetISO(float ISO) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.CameraISO = ISO;
}

float UCameraSensor::GetISO() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.CameraISO;
}

void UCameraSensor::SetAperture(float aperture) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.DepthOfFieldFstop = aperture;
}

float UCameraSensor::GetAperture() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.DepthOfFieldFstop;
}

void UCameraSensor::SetFocalDistance(float Distance) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.DepthOfFieldFocalDistance = Distance;
}

float UCameraSensor::GetFocalDistance() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.DepthOfFieldFocalDistance;
}

void UCameraSensor::SetDepthBlurAmount(float Amount) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.DepthOfFieldDepthBlurAmount = Amount;
}

float UCameraSensor::GetDepthBlurAmount() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.DepthOfFieldDepthBlurAmount;
}

void UCameraSensor::SetDepthBlurRadius(float Radius) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.DepthOfFieldDepthBlurRadius = Radius;
}

float UCameraSensor::GetDepthBlurRadius() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.DepthOfFieldDepthBlurRadius;
}

void UCameraSensor::SetBladeCount(int Count) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.DepthOfFieldBladeCount = Count;
}

int UCameraSensor::GetBladeCount() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.DepthOfFieldBladeCount;
}

void UCameraSensor::SetDepthOfFieldMinFstop(float MinFstop) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.DepthOfFieldMinFstop = MinFstop;
}

float UCameraSensor::GetDepthOfFieldMinFstop() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.DepthOfFieldMinFstop;
}

void UCameraSensor::SetFilmSlope(float Slope) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.FilmSlope = Slope;
}

float UCameraSensor::GetFilmSlope() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.FilmSlope;
}

void UCameraSensor::SetFilmToe(float Toe) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.FilmToe = Toe;
}

float UCameraSensor::GetFilmToe() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.FilmToe;
}

void UCameraSensor::SetFilmShoulder(float Shoulder) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.FilmShoulder = Shoulder;
}

float UCameraSensor::GetFilmShoulder() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.FilmShoulder;
}

void UCameraSensor::SetFilmBlackClip(float BlackClip) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.FilmBlackClip = BlackClip;
}

float UCameraSensor::GetFilmBlackClip() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.FilmBlackClip;
}

void UCameraSensor::SetFilmWhiteClip(float WhiteClip) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.FilmWhiteClip = WhiteClip;
}

float UCameraSensor::GetFilmWhiteClip() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.FilmWhiteClip;
}

void UCameraSensor::SetExposureMinBrightness(float Brightness) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.AutoExposureMinBrightness = Brightness;
}

float UCameraSensor::GetExposureMinBrightness() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.AutoExposureMinBrightness;
}

void UCameraSensor::SetExposureMaxBrightness(float Brightness) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.AutoExposureMaxBrightness = Brightness;
}

float UCameraSensor::GetExposureMaxBrightness() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.AutoExposureMaxBrightness;
}

void UCameraSensor::SetExposureSpeedDown(float Speed) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.AutoExposureSpeedDown = Speed;
}

float UCameraSensor::GetExposureSpeedDown() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.AutoExposureSpeedDown;
}

void UCameraSensor::SetExposureSpeedUp(float Speed) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.AutoExposureSpeedUp = Speed;
}

float UCameraSensor::GetExposureSpeedUp() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.AutoExposureSpeedUp;
}

void UCameraSensor::SetMotionBlurIntensity(float Intensity) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.MotionBlurAmount = Intensity;
}

float UCameraSensor::GetMotionBlurIntensity() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.MotionBlurAmount;
}

void UCameraSensor::SetMotionBlurMaxDistortion(float MaxDistortion) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.MotionBlurMax = MaxDistortion;
}

float UCameraSensor::GetMotionBlurMaxDistortion() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.MotionBlurMax;
}

void UCameraSensor::SetMotionBlurMinObjectScreenSize(float ScreenSize) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.MotionBlurPerObjectSize = ScreenSize;
}

float UCameraSensor::GetMotionBlurMinObjectScreenSize() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.MotionBlurPerObjectSize;
}

void UCameraSensor::SetLensFlareIntensity(float Intensity) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.LensFlareIntensity = Intensity;
}

float UCameraSensor::GetLensFlareIntensity() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.LensFlareIntensity;
}

void UCameraSensor::SetBloomIntensity(float Intensity) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.BloomIntensity = Intensity;
}

float UCameraSensor::GetBloomIntensity() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.BloomIntensity;
}

void UCameraSensor::SetWhiteTemp(float Temp) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.WhiteTemp = Temp;
}

float UCameraSensor::GetWhiteTemp() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.WhiteTemp;
}

void UCameraSensor::SetWhiteTint(float Tint) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.WhiteTint = Tint;
}

float UCameraSensor::GetWhiteTint() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.WhiteTint;
}

void UCameraSensor::SetChromAberrIntensity(float Intensity) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.SceneFringeIntensity = Intensity;
}

float UCameraSensor::GetChromAberrIntensity() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.SceneFringeIntensity;
}

void UCameraSensor::SetChromAberrOffset(float chromAberrOffset) {
	check(SceneCapture != nullptr);
	SceneCapture->PostProcessSettings.ChromaticAberrationStartOffset = chromAberrOffset;
}

float UCameraSensor::GetChromAberrOffset() const {
	check(SceneCapture != nullptr);
	return SceneCapture->PostProcessSettings.ChromaticAberrationStartOffset;
}

void UCameraSensor::EnqueueRenderSceneImmediate() {
	GetCaptureComponent2D()->CaptureScene();
}
