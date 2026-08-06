/*
#include "TaggedComponent.h"


#include "ConstructorHelpers.h"
#include "Holodeck.h"

#include "Rendering/SkeletalMeshRenderData.h"
#include "SkeletalRenderPublic.h"


//
// UTaggedComponent
//
UTaggedComponent::UTaggedComponent(const FObjectInitializer& ObjectInitializer)
: UPrimitiveComponent(ObjectInitializer), Color(1, 1, 1, 1)
{
  FString MaterialPath =
TEXT("Material'/Carla/PostProcessingMaterials/AnnotationColor.AnnotationColor'");
  static ConstructorHelpers::FObjectFinder<UMaterial>
TaggedMaterialObject(*MaterialPath);
  // TODO: Replace with VertexColorViewModeMaterial_ColorOnly?

  TaggedMaterial = TaggedMaterialObject.Object;
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = false;
}

void UTaggedComponent::OnRegister()
{
  Super::OnRegister();

  TaggedMID = UMaterialInstanceDynamic::Create(TaggedMaterial, this,
TEXT("TaggedMaterialMID"));

  if (!IsValid(TaggedMID))
  {
	UE_LOG(LogHolodeck, Error, TEXT("Failed to create MID!"));
  }

  SetColor(Color);
}

void UTaggedComponent::SetColor(FLinearColor NewColor)
{
  Color = NewColor;

  if (IsValid(TaggedMID))
  {
	TaggedMID->SetVectorParameterValue("AnnotationColor", Color);
  }
}

FLinearColor UTaggedComponent::GetColor()
{
  return Color;
}

FBoxSphereBounds UTaggedComponent::CalcBounds(const FTransform & LocalToWorld)
const
{
  USceneComponent * ParentComponent = GetAttachParent();
  if(ParentComponent)
  {
	return ParentComponent->CalcBounds(LocalToWorld);
  }
  return FBoxSphereBounds();
}

FPrimitiveSceneProxy * UTaggedComponent::CreateSceneProxy()
{
  USceneComponent * ParentComponent = GetAttachParent();

  if (!IsValid(ParentComponent))
  {
	UE_LOG(LogHolodeck, Error, TEXT("Invalid parent component"));
	return NULL;
  }

  USkeletalMeshComponent * SkeletalMeshComponent =
Cast<USkeletalMeshComponent>(ParentComponent); UStaticMeshComponent *
StaticMeshComponent = Cast<UStaticMeshComponent>(ParentComponent);
  UHierarchicalInstancedStaticMeshComponent *
HierarchicalInstancedStaticMeshComponent =
	  Cast<UHierarchicalInstancedStaticMeshComponent>(ParentComponent);
  UInstancedStaticMeshComponent* InstancedStaticMeshComponent =
Cast<UInstancedStaticMeshComponent>(ParentComponent); if
(HierarchicalInstancedStaticMeshComponent)
  {
	return CreateSceneProxy(HierarchicalInstancedStaticMeshComponent);
  }
  else if (InstancedStaticMeshComponent)
  {
	return CreateSceneProxy(InstancedStaticMeshComponent);
  }
  else
  if (IsValid(StaticMeshComponent))
  {
	return CreateSceneProxy(StaticMeshComponent);
  }
  else if (IsValid(SkeletalMeshComponent))
  {
	bSkeletalMesh = true;
	return CreateSceneProxy(SkeletalMeshComponent);
  }

  UE_LOG(LogHolodeck, Error, TEXT("Unknown type of parent component: %s"),
*ParentComponent->GetClass()->GetName());

  return NULL;
}

FPrimitiveSceneProxy * UTaggedComponent::CreateSceneProxy(UStaticMeshComponent *
StaticMeshComponent)
{
  // Make sure static mesh has render data
  UStaticMesh * StaticMesh = StaticMeshComponent->GetStaticMesh();

  if (StaticMesh == NULL)
  {
	UE_LOG(LogHolodeck, Error, TEXT("Failed to create scene proxy for static
mesh component (because static mesh is null): %s"),
*StaticMeshComponent->GetReadableName()); return NULL;
  }

  if (StaticMesh->GetRenderData() == NULL)
  {
	UE_LOG(LogHolodeck, Error, TEXT("Failed to create scene proxy for static
mesh component (because render data is null): %s"),
*StaticMeshComponent->GetReadableName()); return NULL;
  }


  if (StaticMesh->GetRenderData()->LODResources.Num() == 0)
  {
	UE_LOG(LogHolodeck, Error, TEXT("Failed to create scene proxy for static
mesh component (because num LOD resources is 0): %s"),
*StaticMeshComponent->GetReadableName()); return NULL;
  }

  return new FTaggedStaticMeshSceneProxy(StaticMeshComponent, true, TaggedMID);
}

FPrimitiveSceneProxy * UTaggedComponent::CreateSceneProxy(USkeletalMeshComponent
* SkeletalMeshComponent)
{
  if (bShouldWaitFrame)
  {
	return nullptr;
  }
  ERHIFeatureLevel::Type SceneFeatureLevel = GetWorld()->FeatureLevel;
		FSkeletalMeshRenderData* SkelMeshRenderData =
SkeletalMeshComponent->GetSkeletalMeshRenderData();

		// Only create a scene proxy for rendering if properly initialized
		if (SkelMeshRenderData &&
				SkelMeshRenderData->LODRenderData.IsValidIndex(SkeletalMeshComponent->PredictedLODLevel)
&& !SkeletalMeshComponent->bHideSkin && SkeletalMeshComponent->MeshObject)
		{
				// Only create a scene proxy if the bone count being used is
supported, or if we don't have a skeleton (this is the case with destructibles)
				int32 MinLODIndex = SkeletalMeshComponent->ComputeMinLOD();
				int32 MaxBonesPerChunk =
SkelMeshRenderData->GetMaxBonesPerSection(MinLODIndex); int32
MaxSupportedNumBones = SkeletalMeshComponent->MeshObject->IsCPUSkinned() ?
MAX_int32 : GetFeatureLevelMaxNumberOfBones(SceneFeatureLevel); if
(MaxBonesPerChunk <= MaxSupportedNumBones)
				{
						return new
FTaggedSkeletalMeshSceneProxy(SkeletalMeshComponent, SkelMeshRenderData,
TaggedMID);
				}
		}
  return nullptr;
}

FPrimitiveSceneProxy *
UTaggedComponent::CreateSceneProxy(UHierarchicalInstancedStaticMeshComponent *
MeshComponent)
{
		// Verify that the mesh is valid before using it.
		const bool bMeshIsValid =
				// make sure we have instances
				(MeshComponent->PerInstanceRenderData.IsValid()) &&
				// make sure we have an actual staticmesh
				MeshComponent->GetStaticMesh() &&
				MeshComponent->GetStaticMesh()->HasValidRenderData(false) &&
				// You really can't use hardware instancing on the consoles with
multiple elements because they share the same index buffer.
				// @todo: Level error or something to let LDs know this
				1;//GetStaticMesh()->LODModels(0).Elements.Num() == 1;

		if (bMeshIsValid)
		{
				bool bIsGrass = !MeshComponent->PerInstanceSMData.Num();
				return new
FTaggedHierarchicalStaticMeshSceneProxy(MeshComponent, bIsGrass,
GetWorld()->FeatureLevel, TaggedMID);
		}
		return nullptr;
}

FPrimitiveSceneProxy *
UTaggedComponent::CreateSceneProxy(UInstancedStaticMeshComponent *
MeshComponent)
{
		// Verify that the mesh is valid before using it.
		const bool bMeshIsValid =
				// make sure we have instances
				(MeshComponent->PerInstanceRenderData.IsValid()) &&
				// make sure we have an actual staticmesh
				MeshComponent->GetStaticMesh() &&
				MeshComponent->GetStaticMesh()->HasValidRenderData(false) &&
				// You really can't use hardware instancing on the consoles with
multiple elements because they share the same index buffer.
				// @todo: Level error or something to let LDs know this
				1;//GetStaticMesh()->LODModels(0).Elements.Num() == 1;

		if (bMeshIsValid)
		{
				return new FTaggedInstancedStaticMeshSceneProxy(MeshComponent,
GetWorld()->FeatureLevel, TaggedMID);
		}
		return nullptr;
}

void UTaggedComponent::TickComponent(float DeltaTime, ELevelTick TickType,
FActorComponentTickFunction * ThisTickFunction)
{
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

  // // TODO: Try removing this
  if (bSkeletalMesh)
  {
	// MarkRenderTransformDirty();
	MarkRenderStateDirty();
	if(bShouldWaitFrame)
	{
	  if(NumFramesToWait < 0)
	  {
		bShouldWaitFrame = false;
	  }
	  NumFramesToWait--;
	}
  }
}
*/
