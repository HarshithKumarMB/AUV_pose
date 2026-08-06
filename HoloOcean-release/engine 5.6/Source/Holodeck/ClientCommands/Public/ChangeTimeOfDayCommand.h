#pragma once

#include "Command.h"
#include "Holodeck.h"

#include "ChangeTimeOfDayCommand.generated.h"

/**
 * ChangeTimeOfDayCommand
 * Command used to change the world's time of day
 * NumParams expects one argument representing the time of day desired (a number
 * between 0 and 23 inclusive)
 */

UCLASS()
class HOLODECK_API UChangeTimeOfDayCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};