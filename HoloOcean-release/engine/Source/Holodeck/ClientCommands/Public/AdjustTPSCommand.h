#pragma once

#include "Holodeck.h"
#include "Command.h"
#include "HolodeckSensor.h"

#include "AdjustTPSCommand.generated.h"

/**
 * AdjustTPSCommand
 * Command to adjust tps of the simulation.
 *
 * NumberParameters expect 1 argument, representing the ticks per sec
 */
UCLASS()
class HOLODECK_API UAdjustTPSCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
