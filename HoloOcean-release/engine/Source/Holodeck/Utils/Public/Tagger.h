// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
#pragma once
#include "ObjectLabel.h"
#include "Tagger.generated.h"
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class HOLODECK_API ATagger : public AActor {
	GENERATED_BODY()

public:
	ATagger();

	/// Set the tag of an actor.
	///
	/// If bTagForSemanticSegmentation true, activate the custom depth pass. This
	/// pass is necessary for rendering the semantic segmentation. However, it may
	/// add a performance penalty since occlusion doesn't seem to be applied to
	/// objects having this value active.
	static void TagActor(const AActor& Actor, bool bShouldTagForSemanticSegmentation);

	/// Set the tag of every actor in level.
	///
	/// If shouldTagForSemanticSegmentation true, activate the custom depth pass.
	/// This pass is necessary for rendering the semantic segmentation. However,
	/// it may add a performance penalty since occlusion doesn't seem to be
	/// applied to objects having this value active.
	static void TagActorsInLevel(UWorld& World, bool bShouldTagForSemanticSegmentation);

	static void TagActorsInLevel(ULevel& Level, bool bShouldTagForSemanticSegmentation);

	// Retire the tag of an already tagged component
	static holoocean::ObjectLabel
	GetTagOfTaggedComponent(const UPrimitiveComponent& Component) {
		return static_cast<holoocean::ObjectLabel>(Component.CustomDepthStencilValue);
	}

	/// Retrieve the tags of an already tagged actor. CityObjectLabel::None is
	/// not added to the array.
	static void
	GetTagsOfTaggedActor(const AActor& Actor, TSet<holoocean::ObjectLabel>& Tags);

	/// Return true if @a Component has been tagged with the given @a Tag.
	static bool
	MatchComponent(const UPrimitiveComponent& Component, holoocean::ObjectLabel Tag) {
		return (Tag == GetTagOfTaggedComponent(Component));
	}

	/// Retrieve the tags of an already tagged actor. CityObjectLabel::None is
	/// not added to the array.
	static FString GetTagAsString(holoocean::ObjectLabel Tag);

	/// Method that computes the label corresponding to a folder path
	static holoocean::ObjectLabel GetLabelByFolderName(const FString& String);

	virtual void BeginPlay() override {
		if (UWorld* world = GetWorld(); world) {
			TagActorsInLevel(*world, true);
		}
	}

	/// Method that computes the label corresponding to an specific object
	/// using the folder path in which it is stored
	template <typename T>
	static holoocean::ObjectLabel GetLabelByPath(const TObjectPtr<T> Object) {
		const FString	Path = Object->GetPathName();
		TArray<FString> StringArray;
		Path.ParseIntoArray(StringArray, TEXT("/"), false);
		UE_LOG(
			LogHolodeck,
			Log,
			TEXT("Raw Path: %s"),
			*Object
				 ->GetPathName()); // LogHolodeck: Raw Path:
								   // /Engine/BasicShapes/Plane.Plane //it's getting the
								   // wrong path, not the level path but the asset path

		if (StringArray.Num() > 4) {
			const FString&		   FolderName = StringArray[StringArray.Num() - 3];
			holoocean::ObjectLabel folderLabel =
				GetLabelByFolderName(StringArray[StringArray.Num() - 3]);
			return folderLabel;
		}
		return holoocean::ObjectLabel::None;
	}

	static holoocean::ObjectLabel
	GetLabelBylevelPath(const TObjectPtr<UObject> Object) {
		if (!Object)
			return holoocean::ObjectLabel::None;
		// UE_LOG(LogHolodeck, Warning, TEXT("Object class: %s | name: %s"),
		// *Object->GetClass()->GetName(), *Object->GetName());
		FName FolderPath;

// IMPORTANT: can only access the GetFolderPath() function from within the editor!!!!!!!
// so if you are trying to tag objects using folder names, make sure the tagger.cpp is
// in your level then click on the tagger from the outliner and select "Trigger Tag
// Objects"
#if WITH_EDITOR
		// Find folder path for actors
		if (const AActor* Actor = Cast<AActor>(Object)) {
			FolderPath = Actor->GetFolderPath();
		}
		// Find folder path for actor components
		else if (const UActorComponent* Component = Cast<UActorComponent>(Object)) {
			if (const AActor* Owner = Component->GetOwner()) {
				FolderPath = Owner->GetFolderPath();
			}
		}

		// Tag returned is the last folder in the path
		if (!FolderPath.IsNone()) {
			// UE_LOG(LogHolodeck, Log, TEXT("Level folder path: %s"),
			// *FolderPath.ToString());

			// grab just the last folder in the path:
			TArray<FString> Parts;
			FolderPath.ToString().ParseIntoArray(Parts, TEXT("/"), true);
			if (Parts.Num() > 0) {
				// UE_LOG(LogHolodeck, Error, TEXT("FOLDER TAG FOUND!: %s"),
				// *Parts.Last());
				return GetLabelByFolderName(Parts.Last());
			}
		}
#endif

		return holoocean::ObjectLabel::None;
	}

	static void AddActorTagToAsset(
		UPrimitiveComponent&		  Component,
		const holoocean::ObjectLabel& Label);

	static void SetStencilValue(
		UPrimitiveComponent&		  Component,
		const holoocean::ObjectLabel& Label,
		const bool					  bShouldSetRenderCustomDepth);

	static FLinearColor
	GetActorLabelColor(const AActor& Actor, const holoocean::ObjectLabel& Label);

	static bool IsThing(const holoocean::ObjectLabel& Label);

protected:
#if WITH_EDITOR
	virtual void
	PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif // WITH_EDITOR

private:
	UPROPERTY(Category = "Tagger", EditAnywhere)
	bool bTriggerTagObjects = false;

	UPROPERTY(Category = "Tagger", EditAnywhere)
	bool bShouldTagForSemanticSegmentation = false;
};