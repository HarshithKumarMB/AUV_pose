#pragma once

#include "Holodeck.h"

#include <map>
#include <vector>

#include "AddSensorCommand.h"
#include "AdjustFPSCommand.h"
#include "AdjustRenderQualityCommand.h"
#include "AdjustTPSCommand.h"
#include "AirFogCommand.h"
#include "ChangeTimeOfDayCommand.h"
#include "ChangeWeatherCommand.h"
#include "Command.h"
#include "CustomCommand.h"
#include "DebugDrawCommand.h"
#include "OceanCurrentsCommand.h"
#include "RGBCameraRateCommand.h"
#include "RemoveSensorCommand.h"
#include "RenderViewportCommand.h"
#include "RotateSensorCommand.h"
#include "SendAcousticMessageCommand.h"
#include "SendOpticalMessageCommand.h"
#include "SetRainParametersCommand.h"
#include "SpawnAgentCommand.h"
#include "TeleportCameraCommand.h"
#include "TideCommand.h"
#include "TurnOffFlashlightCommand.h"
#include "TurnOnFlashlightCommand.h"
#include "WaterColorCommand.h"
#include "WaterFogCommand.h"

#include "CommandFactory.generated.h"

class AHolodeckGameMode;

/**
 * UCommandFactory
 * This is the class that should be used to instantiate UCommand objects. Feed
 * it the name of the command along with the parameters that the command will
 * need to execute. If the parameters are not needed, then give it nullptr or
 * empty vectors and it will work fine.
 * The purpose of this was to separate knowledge of specific commands from the
 * command center, to remove circular dependencies, and to give an easy way of
 * spawning commands. When you make a new command, make sure to add it to the
 * CommandMap in the MakeCommand function in the cpp file
 */
UCLASS(ClassGroup = (Custom), abstract)
class HOLODECK_API UCommandFactory : public UObject {
	GENERATED_BODY()

	typedef std::map<std::string, UCommand* (*)()> UCommandMapType;

public:
	/**
	 * MakeCommand
	 * This is the factory method for producing commands.
	 */
	static UCommand* MakeCommand(
		const std::string&				Name,
		const std::vector<float>&		NumberParameters,
		const std::vector<std::string>& StringParameters,
		AActor*							ParameterGameMode);

private:
	/**
	 * UCommandFactory
	 * Default constructor. Should not be instantiated, hence it is private.
	 */
	UCommandFactory() {};

	template <typename T> static UCommand* CreateInstance() { return NewObject<T>(); }
};
