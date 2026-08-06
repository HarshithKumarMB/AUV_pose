#pragma once

#include "Holodeck.h"

#include "AbuseSensor.h"
#include "AcousticBeaconSensor.h"
#include "BSTSensor.h"
#include "CameraSensor.h"
#include "CleanUpTask.h"
#include "CollisionSensor.h"
#include "CupGameTask.h"
#include "DVLSensor.h"
#include "DepthSensor.h"
#include "DistanceTask.h"
#include "DynamicsSensor.h"
#include "FollowTask.h"
#include "GPSSensor.h"
#include "IMUSensor.h"
#include "ImagingSonar.h"
#include "JointRotationSensor.h"
#include "LocationSensor.h"
#include "LocationTask.h"
#include "MagnetometerSensor.h"
#include "OpticalModemSensor.h"
#include "OrientationSensor.h"
#include "PressureSensor.h"
#include "RangeFinderSensor.h"
#include "PoseSensor.h"
#include "ProfilingSonar.h"
#include "CameraSensor.h"
#include "RGBCamera.h"
#include "DepthCamera.h"
#include "SemanticSegmentationCamera.h"
#include "RaycastLidar.h"
#include "RaycastSemanticLidar.h"
#include "RelativeSkeletalPositionSensor.h"
#include "RotationSensor.h"
#include "SemanticSegmentationCamera.h"
#include "SidescanSonar.h"
#include "SinglebeamSonar.h"
#include "VelocitySensor.h"
#include "ViewportCapture.h"
#include "WorldNumSensor.h"

#include <map>

#include "Command.h"

#include "AddSensorCommand.generated.h"

/**
 * AddSensorCommand
 * Command used to add a sensor to an agent
 * Warning: This command is meant for initialization. Adding a sensor with the
 * same name as a previously removed sensor may cause problems.
 *
 * StringParameters expects five arguments: the agent name, sensor name, sensor
 * class, sensor parameters, and socket. NumberParameters expects six arguments:
 * locations x, y, and z and rotations pitch, yaw, and roll.
 */
UCLASS()
class HOLODECK_API UAddSensorCommand : public UCommand {
	GENERATED_BODY()

	typedef std::map<FString, UClass*> USensorMapType;

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
