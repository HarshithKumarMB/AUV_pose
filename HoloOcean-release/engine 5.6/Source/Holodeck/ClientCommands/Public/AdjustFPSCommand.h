#pragma once

#include "Holodeck.h"
#include "Command.h"
#include "HolodeckSensor.h"

#include "AdjustFPSCommand.generated.h"

/**
 * AdjustFPSCommand
 * Command to adjust fps of the simulation.
 *
 * NumberParameters expect 1 argument, representing the fps
 */
UCLASS()
class HOLODECK_API UAdjustFPSCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
