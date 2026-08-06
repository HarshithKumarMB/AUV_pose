#pragma once

#include "Holodeck.h"

#include "Command.h"

#include "TideCommand.generated.h"

/**
 * Command to change the tides (surface level)
 *
 *
 * NumberParams expects two arguments representing either the amount to change
 * the surface level by and a 0 or the new surface level and a 1
 */
UCLASS()
class HOLODECK_API UTideCommand : public UCommand {
	GENERATED_BODY()

public:
	static FVector PPVUnderwater;
	static FVector PPVAir;
	static bool	   waterppvfirst;
	static bool	   airppvfirst;
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
