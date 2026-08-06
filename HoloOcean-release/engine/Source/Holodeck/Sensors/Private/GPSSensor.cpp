// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "GPSSensor.h"
#include "HolodeckBuoyantAgent.h"
#include "Holodeck.h"

UGPSSensor::UGPSSensor() {
	PrimaryComponentTick.bCanEverTick = true;
	SensorName = "GPSSensor";
}

void UGPSSensor::ParseSensorParms(FString ParmsJson) {
	Super::ParseSensorParms(ParmsJson);

	TSharedPtr<FJsonObject>		   JsonParsed;
	TSharedRef<TJsonReader<TCHAR>> JsonReader =
		TJsonReaderFactory<TCHAR>::Create(ParmsJson);
	if (FJsonSerializer::Deserialize(JsonReader, JsonParsed)) {

		if (JsonParsed->HasTypedField<EJson::Number>("Sigma")) {
			mvn.initSigma(JsonParsed->GetNumberField("Sigma"));
		}
		if (JsonParsed->HasTypedField<EJson::Array>("Sigma")) {
			mvn.initSigma(JsonParsed->GetArrayField("Sigma"));
		}

		if (JsonParsed->HasTypedField<EJson::Number>("Cov")) {
			mvn.initCov(JsonParsed->GetNumberField("Cov"));
		}
		if (JsonParsed->HasTypedField<EJson::Array>("Cov")) {
			mvn.initCov(JsonParsed->GetArrayField("Cov"));
		}

		if (JsonParsed->HasTypedField<EJson::Number>("Depth")) {
			// Take absolute value in case people put it in as a negative (ie depth is
			// down)
			GPSDepth = UKismetMathLibrary::Abs(JsonParsed->GetNumberField("Depth"));
		}

		if (JsonParsed->HasTypedField<EJson::Number>("DepthSigma")) {
			depthMVN.initSigma(JsonParsed->GetNumberField("DepthSigma"));
		}
		if (JsonParsed->HasTypedField<EJson::Number>("DepthCov")) {
			depthMVN.initCov(JsonParsed->GetNumberField("DepthCov"));
		}
	} else {
		UE_LOG(
			LogHolodeck,
			Fatal,
			TEXT("UGPSSensor::ParseSensorParms:: Unable to parse json."));
	}
}

void UGPSSensor::InitializeSensor() {
	Super::InitializeSensor();

	// You need to get the pointer to the object you are attached to.
	Parent = this->GetAttachParent();
	// Need to get the agent you are attached to for receiving a GPS measurement with
	// changing water level
	if (Parent) {
		Agent = Parent->GetOwner();
	}
	if (Agent) {
		BuoyantAgent = Cast<AHolodeckBuoyantAgent>(Agent);
	}
}

void UGPSSensor::TickSensorComponent(
	float						 DeltaTime,
	ELevelTick					 TickType,
	FActorComponentTickFunction* ThisTickFunction) {
	// check if your parent pointer is valid, and if the sensor is on. Then get
	// the location and buffer, then send the location to the buffer.
	if (Parent != nullptr && bOn) {
		FVector Location = this->GetComponentLocation();

		// Get location
		float* FloatBuffer = static_cast<float*>(Buffer);
		Location = ConvertLinearVector(Location, UEToClient);
		float depth = -1 * Location.Z + depthMVN.sampleFloat();
		if (BuoyantAgent == nullptr) {
			UE_LOG(
				LogHolodeck,
				Log,
				TEXT(
					"BuoyantAgent is a nullptr -- GPSSensor does not account for changing tide"));
		} else {
			depth += (BuoyantAgent->SurfaceLevel / 100);
		}
		if (depth < GPSDepth) {
			Location += mvn.sampleFVector();
			FloatBuffer[0] = Location.X;
			FloatBuffer[1] = Location.Y;
			FloatBuffer[2] = Location.Z;
		} else {
			FloatBuffer[0] = std::numeric_limits<double>::quiet_NaN();
			FloatBuffer[1] = std::numeric_limits<double>::quiet_NaN();
			FloatBuffer[2] = std::numeric_limits<double>::quiet_NaN();
		}
	}
}
