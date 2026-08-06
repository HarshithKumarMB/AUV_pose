#include "OceanCurrentsCommand.h"
#include "Holodeck.h"
#include "HolodeckBuoyantAgent.h"
#include "HolodeckGameMode.h"

void UOceanCurrentsCommand::Execute() {
	if (StringParams.size() != 1 || NumberParams.size() != 4) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT("Unexpected argument length found in v. Command not executed."));
		return;
	}

	AHolodeckGameMode* GameTarget = static_cast<AHolodeckGameMode*>(Target);
	if (GameTarget == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UCommand::Target is not a UHolodeckGameMode*. "
				"UAddSensorCommand::Sensor not added."));
		return;
	}

	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UAddSensorCommand::Execute found world as nullptr. Sensor not "
				"added."));
		return;
	}

	FString AgentName = StringParams[0].c_str();
	float	Ocean_Current_Velocity_X = NumberParams[0];
	float	Ocean_Current_Velocity_Y = NumberParams[1];
	float	Ocean_Current_Velocity_Z = NumberParams[2];

	AHolodeckAgent* Agent = GetAgent(AgentName);

	AHolodeckBuoyantAgent* BuoyantAgent = static_cast<AHolodeckBuoyantAgent*>(Agent);

	BuoyantAgent->Ocean_Current_Velocity_X = Ocean_Current_Velocity_X;
	BuoyantAgent->Ocean_Current_Velocity_Y = Ocean_Current_Velocity_Y;
	BuoyantAgent->Ocean_Current_Velocity_Z = Ocean_Current_Velocity_Z;

	int Ocean_Current_Vehicle_Debugging = NumberParams[3];
	BuoyantAgent->Ocean_Current_Vehicle_Debugging = Ocean_Current_Vehicle_Debugging;
}
