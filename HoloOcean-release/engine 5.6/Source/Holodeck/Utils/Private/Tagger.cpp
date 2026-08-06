#include "Tagger.h"

template <typename T> static auto CastEnum(T label) {
	return static_cast<typename std::underlying_type_t<T>>(label);
}

holoocean::ObjectLabel ATagger::GetLabelByFolderName(const FString& String) {
	if (String == "Cube")
		return holoocean::ObjectLabel::Cube;
	else if (String == "Sphere")
		return holoocean::ObjectLabel::Sphere;
	else if (String == "BaseShape")
		return holoocean::ObjectLabel::BaseShape;
	else if (String == "Landscape")
		return holoocean::ObjectLabel::Landscape;
	else if (String == "GroundGrass")
		return holoocean::ObjectLabel::GroundGrass;
	else if (String == "GroundRock")
		return holoocean::ObjectLabel::GroundRock;
	else if (String == "Ground")
		return holoocean::ObjectLabel::Ground;
	else if (String == "GroundPath")
		return holoocean::ObjectLabel::GroundPath;
	else if (String == "WaterPlane")
		return holoocean::ObjectLabel::WaterPlane;
	else if (String == "Boat")
		return holoocean::ObjectLabel::Boat;
	else if (String == "Yacht")
		return holoocean::ObjectLabel::Yacht;
	else if (String == "ContainerBoat")
		return holoocean::ObjectLabel::ContainerBoat;
	else if (String == "Concrete")
		return holoocean::ObjectLabel::Concrete;
	else if (String == "Pipe")
		return holoocean::ObjectLabel::Pipe;
	else if (String == "PipeCover")
		return holoocean::ObjectLabel::PipeCover;
	else if (String == "VentCover")
		return holoocean::ObjectLabel::VentCover;
	else if (String == "Rock")
		return holoocean::ObjectLabel::Rock;
	else if (String == "Seaweed")
		return holoocean::ObjectLabel::Seaweed;
	else if (String == "Coral")
		return holoocean::ObjectLabel::Coral;
	else if (String == "Plane")
		return holoocean::ObjectLabel::Plane;
	else if (String == "Sub")
		return holoocean::ObjectLabel::Sub;
	else if (String == "Pier")
		return holoocean::ObjectLabel::Pier;
	else if (String == "Buoy")
		return holoocean::ObjectLabel::Buoy;
	else if (String == "Trash")
		return holoocean::ObjectLabel::Trash;
	else if (String == "Grass")
		return holoocean::ObjectLabel::Grass;
	else if (String == "Asphalt")
		return holoocean::ObjectLabel::Asphalt;
	else if (String == "Bench")
		return holoocean::ObjectLabel::Bench;
	else if (String == "BikeRack")
		return holoocean::ObjectLabel::BikeRack;
	else if (String == "Building")
		return holoocean::ObjectLabel::Building;
	else if (String == "Bus")
		return holoocean::ObjectLabel::Bus;
	else if (String == "Bush")
		return holoocean::ObjectLabel::Bush;
	else if (String == "Car")
		return holoocean::ObjectLabel::Car;
	else if (String == "Ceiling")
		return holoocean::ObjectLabel::Ceiling;
	else if (String == "Chair")
		return holoocean::ObjectLabel::Chair;
	else if (String == "Cone")
		return holoocean::ObjectLabel::Cone;
	else if (String == "Crate")
		return holoocean::ObjectLabel::Crate;
	else if (String == "Desk")
		return holoocean::ObjectLabel::Desk;
	else if (String == "Dumpster")
		return holoocean::ObjectLabel::Dumpster;
	else if (String == "FireHydrant")
		return holoocean::ObjectLabel::FireHydrant;
	else if (String == "Floor")
		return holoocean::ObjectLabel::Floor;
	else if (String == "GarbageCan")
		return holoocean::ObjectLabel::GarbageCan;
	else if (String == "Pallet")
		return holoocean::ObjectLabel::Pallet;
	else if (String == "ParkingGate")
		return holoocean::ObjectLabel::ParkingGate;
	else if (String == "PatioUmbrella")
		return holoocean::ObjectLabel::PatioUmbrella;
	else if (String == "Railing")
		return holoocean::ObjectLabel::Railing;
	else if (String == "SemiTruck")
		return holoocean::ObjectLabel::SemiTruck;
	else if (String == "Sidewalk")
		return holoocean::ObjectLabel::Sidewalk;
	else if (String == "SpeedLimitSign")
		return holoocean::ObjectLabel::SpeedLimitSign;
	else if (String == "StopSign")
		return holoocean::ObjectLabel::StopSign;
	else if (String == "StreetLamps")
		return holoocean::ObjectLabel::StreetLamps;
	else if (String == "Table")
		return holoocean::ObjectLabel::Table;
	else if (String == "Tree")
		return holoocean::ObjectLabel::Tree;
	else if (String == "Wall")
		return holoocean::ObjectLabel::Wall;
	else if (String == "Unlabeled")
		return holoocean::ObjectLabel::Unlabeled;
	else
		return holoocean::ObjectLabel::None;
}

void ATagger::AddActorTagToAsset(
	UPrimitiveComponent&		  Component,
	const holoocean::ObjectLabel& Label) {
	Component.ComponentTags.Add(FName(GetTagAsString(Label)));
}

void ATagger::SetStencilValue(
	UPrimitiveComponent&		  Component,
	const holoocean::ObjectLabel& Label,
	const bool					  bShouldSetRenderCustomDepth) {
	Component.SetCustomDepthStencilValue(CastEnum(Label));
	Component.SetRenderCustomDepth(
		bShouldSetRenderCustomDepth && (Label != holoocean::ObjectLabel::None));
}

// Carla's implementation looks for all things like pedestrians, traffic signs,
// etc. instead
bool ATagger::IsThing(const holoocean::ObjectLabel& Label) {
	return (Label != holoocean::ObjectLabel::None);
}

FLinearColor
ATagger::GetActorLabelColor(const AActor& Actor, const holoocean::ObjectLabel& Label) {
	uint32 id = Actor.GetUniqueID();

	FLinearColor Color(0.0f, 0.0f, 0.0f, 1.0f);
	Color.R = CastEnum(Label) / 255.0f;
	Color.G = ((id & 0x00ff) >> 0) / 255.0f;
	Color.B = ((id & 0xff00) >> 8) / 255.0f;

	return Color;
}

void ATagger::TagActor(const AActor& Actor, bool bShouldTagForSemanticSegmentation) {
	TArray<UStaticMeshComponent*> StaticMeshComponents;
	Actor.GetComponents<UStaticMeshComponent>(StaticMeshComponents);

	for (UStaticMeshComponent* Component : StaticMeshComponents) {
		// will prioritize the actor tag before the folder name
		// if no actor tag found, will use the folder name as tag
		if (Actor.Tags.Num() > 0) {
			auto Label = GetLabelByFolderName(Actor.Tags[0].ToString());
			// UE_LOG(LogHolodeck, Error, TEXT("ACTOR TAG FOUND!: %s"),
			// *Actor.Tags[0].ToString());
			SetStencilValue(*Component, Label, bShouldTagForSemanticSegmentation);
			AddActorTagToAsset(*Component, Label);

			if (!Component->IsVisible() || !Component->GetStaticMesh()) {
				continue;
			}
		} else {
			auto Label = GetLabelBylevelPath(Component);
			SetStencilValue(*Component, Label, bShouldTagForSemanticSegmentation);
			AddActorTagToAsset(*Component, Label);

			if (!Component->IsVisible() || !Component->GetStaticMesh()) {
				continue;
			}
		}
	}
}

void ATagger::TagActorsInLevel(UWorld& World, bool bShouldTagForSemanticSegmentation) {
	for (TActorIterator<AActor> it(&World); it; ++it) {
		TagActor(**it, bShouldTagForSemanticSegmentation);
	}
}

void ATagger::TagActorsInLevel(ULevel& Level, bool bShouldTagForSemanticSegmentation) {
	for (AActor* Actor : Level.Actors) {
		TagActor(*Actor, bShouldTagForSemanticSegmentation);
	}
}

void ATagger::GetTagsOfTaggedActor(
	const AActor&				  Actor,
	TSet<holoocean::ObjectLabel>& Tags) {
	TArray<UPrimitiveComponent*> Components;
	Actor.GetComponents<UPrimitiveComponent>(Components);

	for (auto* Component : Components) {
		if (Component != nullptr) {
			const auto Tag = GetTagOfTaggedComponent(*Component);
			if (Tag != holoocean::ObjectLabel::None) {
				Tags.Add(Tag);
			}
		}
	}
}

FString ATagger::GetTagAsString(holoocean::ObjectLabel Tag) {
	switch (Tag) {
#define HOLOOCEAN_GET_LABEL_STR(lbl)  \
	case holoocean::ObjectLabel::lbl: \
		return TEXT(#lbl);
		default:
			HOLOOCEAN_GET_LABEL_STR(None)
			HOLOOCEAN_GET_LABEL_STR(Cube)
			HOLOOCEAN_GET_LABEL_STR(Sphere)
			HOLOOCEAN_GET_LABEL_STR(BaseShape)
			HOLOOCEAN_GET_LABEL_STR(Landscape)
			HOLOOCEAN_GET_LABEL_STR(GroundGrass)
			HOLOOCEAN_GET_LABEL_STR(GroundRock)
			HOLOOCEAN_GET_LABEL_STR(Ground)
			HOLOOCEAN_GET_LABEL_STR(GroundPath)
			HOLOOCEAN_GET_LABEL_STR(WaterPlane)
			HOLOOCEAN_GET_LABEL_STR(Boat)
			HOLOOCEAN_GET_LABEL_STR(Yacht)
			HOLOOCEAN_GET_LABEL_STR(ContainerBoat)
			HOLOOCEAN_GET_LABEL_STR(Concrete)
			HOLOOCEAN_GET_LABEL_STR(Pipe)
			HOLOOCEAN_GET_LABEL_STR(PipeCover)
			HOLOOCEAN_GET_LABEL_STR(VentCover)
			HOLOOCEAN_GET_LABEL_STR(Rock)
			HOLOOCEAN_GET_LABEL_STR(Seaweed)
			HOLOOCEAN_GET_LABEL_STR(Coral)
			HOLOOCEAN_GET_LABEL_STR(Plane)
			HOLOOCEAN_GET_LABEL_STR(Sub)
			HOLOOCEAN_GET_LABEL_STR(Pier)
			HOLOOCEAN_GET_LABEL_STR(Buoy)
			HOLOOCEAN_GET_LABEL_STR(Trash)
			HOLOOCEAN_GET_LABEL_STR(Grass)
			HOLOOCEAN_GET_LABEL_STR(Asphalt)
			HOLOOCEAN_GET_LABEL_STR(Bench)
			HOLOOCEAN_GET_LABEL_STR(BikeRack)
			HOLOOCEAN_GET_LABEL_STR(Building)
			HOLOOCEAN_GET_LABEL_STR(Bus)
			HOLOOCEAN_GET_LABEL_STR(Bush)
			HOLOOCEAN_GET_LABEL_STR(Car)
			HOLOOCEAN_GET_LABEL_STR(Ceiling)
			HOLOOCEAN_GET_LABEL_STR(Chair)
			HOLOOCEAN_GET_LABEL_STR(Cone)
			HOLOOCEAN_GET_LABEL_STR(Crate)
			HOLOOCEAN_GET_LABEL_STR(Desk)
			HOLOOCEAN_GET_LABEL_STR(Dumpster)
			HOLOOCEAN_GET_LABEL_STR(FireHydrant)
			HOLOOCEAN_GET_LABEL_STR(Floor)
			HOLOOCEAN_GET_LABEL_STR(GarbageCan)
			HOLOOCEAN_GET_LABEL_STR(Pallet)
			HOLOOCEAN_GET_LABEL_STR(ParkingGate)
			HOLOOCEAN_GET_LABEL_STR(PatioUmbrella)
			HOLOOCEAN_GET_LABEL_STR(Railing)
			HOLOOCEAN_GET_LABEL_STR(SemiTruck)
			HOLOOCEAN_GET_LABEL_STR(Sidewalk)
			HOLOOCEAN_GET_LABEL_STR(SpeedLimitSign)
			HOLOOCEAN_GET_LABEL_STR(StopSign)
			HOLOOCEAN_GET_LABEL_STR(StreetLamps)
			HOLOOCEAN_GET_LABEL_STR(Table)
			HOLOOCEAN_GET_LABEL_STR(Tree)
			HOLOOCEAN_GET_LABEL_STR(Wall)
			HOLOOCEAN_GET_LABEL_STR(Unlabeled)
			HOLOOCEAN_GET_LABEL_STR(Any)

#undef HOLOOCEAN_GET_LABEL_STR
	}
}

#if WITH_EDITOR
void ATagger::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
	Super::PostEditChangeProperty(PropertyChangedEvent);
	if (PropertyChangedEvent.Property) {
		if (bTriggerTagObjects && (GetWorld() != nullptr)) {
			TagActorsInLevel(*GetWorld(), bShouldTagForSemanticSegmentation);
		}
	}
	bTriggerTagObjects = false;
}
#endif // WITH_EDITOR

ATagger::ATagger() {
	/*
  if (GetWorld())
  {
		  UE_LOG(LogHolodeck, Warning, TEXT("Tagging all actors in level"));

		  TagActorsInLevel(*GetWorld(), true);
  }
  */
}
