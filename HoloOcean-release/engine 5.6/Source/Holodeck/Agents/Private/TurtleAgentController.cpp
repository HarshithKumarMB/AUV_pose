// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "TurtleAgentController.h"
#include "Holodeck.h"

ATurtleAgentController::ATurtleAgentController(
	const FObjectInitializer& ObjectInitializer)
	: AHolodeckPawnController(ObjectInitializer) {
	UE_LOG(LogTemp, Warning, TEXT("TurtleAgent Controller Initialized"));
}

ATurtleAgentController::~ATurtleAgentController() {}
