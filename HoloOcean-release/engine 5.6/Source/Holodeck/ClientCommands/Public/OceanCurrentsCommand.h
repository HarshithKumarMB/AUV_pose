#pragma once
#include "Command.h"
#include "Holodeck.h"

#include "OceanCurrentsCommand.generated.h"

UCLASS()
class HOLODECK_API UOceanCurrentsCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
