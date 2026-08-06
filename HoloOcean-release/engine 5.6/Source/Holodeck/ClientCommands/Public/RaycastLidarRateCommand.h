#pragma once

#include "Holodeck.h"

#include "Command.h"
#include "RaycastLidarRateCommand.generated.h"

/**
 * RaycastLidarRateCommand
 *
 */
UCLASS(ClassGroup = (Custom))
class HOLODECK_API URaycastLidarRateCommand : public UCommand {
	GENERATED_BODY()
public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
