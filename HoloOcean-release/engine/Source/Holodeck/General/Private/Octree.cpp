// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "Octree.h"
#include <iostream>

// Initialize static variables
// Used when making octree
TArray<FVector> Octree::corners = { FVector(1, 1, 1), // clang-format off
									FVector(1, 1, -1),
									FVector(1, -1, 1),	
									FVector(1, -1, -1),
									FVector(-1, 1, 1),	
									FVector(-1, 1, -1),
									FVector(-1, -1, 1), 
									FVector(-1, -1, -1) };				 // clang-format on
TArray<FVector> Octree::sides = { FVector(0, 0, 1), // clang-format off
								  FVector(0, 0, -1),
								  FVector(0, 1, 0), 
								  FVector(0, -1, 0),
								  FVector(1, 0, 0), 
								  FVector(-1, 0, 0) }; // clang-format on
float Octree::cornerSize = 0.01;

FCollisionQueryParams Octree::params = Octree::init_params();

// Misc constants
float	Octree::OctreeRoot;
float	Octree::OctreeMax;
float	Octree::OctreeMin;
FVector Octree::EnvMin;
FVector Octree::EnvMax;
FVector Octree::EnvCenter;
UWorld* Octree::World;

TDiscardableKeyValueCache<FString, float> Octree::materials;

float sign(float val) {
	bool s = signbit(val);
	if (s)
		return -1.0;
	else
		return 1.0;
}

void Octree::initOctree(UWorld* w) {
	World = w;

	// Load environment size
	if (!FParse::Value(FCommandLine::Get(), TEXT("EnvMinX="), EnvMin.X))
		EnvMin.X = -10;
	if (!FParse::Value(FCommandLine::Get(), TEXT("EnvMinY="), EnvMin.Y))
		EnvMin.Y = -10;
	if (!FParse::Value(FCommandLine::Get(), TEXT("EnvMinZ="), EnvMin.Z))
		EnvMin.Z = -10;
	if (!FParse::Value(FCommandLine::Get(), TEXT("EnvMaxX="), EnvMax.X))
		EnvMax.X = 10;
	if (!FParse::Value(FCommandLine::Get(), TEXT("EnvMaxY="), EnvMax.Y))
		EnvMax.Y = 10;
	if (!FParse::Value(FCommandLine::Get(), TEXT("EnvMaxZ="), EnvMax.Z))
		EnvMax.Z = 10;
	// Clean environment size
	EnvMin = ConvertLinearVector(EnvMin, ClientToUE);
	EnvMax = ConvertLinearVector(EnvMax, ClientToUE);
	FVector min = FVector(
		(int)FGenericPlatformMath::Min(EnvMin.X, EnvMax.X),
		(int)FGenericPlatformMath::Min(EnvMin.Y, EnvMax.Y),
		(int)FGenericPlatformMath::Min(EnvMin.Z, EnvMax.Z));
	FVector max = FVector(
		(int)FGenericPlatformMath::Max(EnvMin.X, EnvMax.X),
		(int)FGenericPlatformMath::Max(EnvMin.Y, EnvMax.Y),
		(int)FGenericPlatformMath::Max(EnvMin.Z, EnvMax.Z));
	EnvMin = min;
	EnvMax = max;
	UE_LOG(LogHolodeck, Log, TEXT("Octree:: EnvMin: %s"), *EnvMin.ToString());
	UE_LOG(LogHolodeck, Log, TEXT("Octree:: EnvMax: %s"), *EnvMax.ToString());

	// Get octree min/max
	float tempVal;
	if (!FParse::Value(FCommandLine::Get(), TEXT("OctreeMin="), tempVal))
		tempVal = .1;
	OctreeMin = (tempVal * 100);
	if (!FParse::Value(FCommandLine::Get(), TEXT("OctreeMax="), tempVal))
		tempVal = 5;
	OctreeMax = (tempVal * 100);

	// Calculate where/how big is biggest octree
	EnvCenter = (EnvMax + EnvMin) / 2;
	OctreeRoot = (EnvMax - EnvMin).GetAbsMax();

	// Make max/root a multiple of min
	tempVal = OctreeMin;
	while (tempVal < OctreeMax) {
		tempVal *= 2;
	}
	OctreeMax = tempVal;
	while (tempVal < OctreeRoot) {
		tempVal *= 2;
	}
	OctreeRoot = tempVal;
	UE_LOG(
		LogHolodeck,
		Log,
		TEXT("Octree:: OctreeMin: %f, OctreeMax: %f, OctreeRoot: %f"),
		OctreeMin,
		OctreeMax,
		OctreeRoot);

	// Load material lookup table
	FString			filePath = FPaths::ProjectDir() + "../../materials.csv";
	TArray<FString> lines;
	FFileHelper::LoadANSITextFileToStrings(*filePath, NULL, lines);
	for (int i = 1; i < lines.Num(); i++) {
		// Split line into elements
		TArray<FString> stringArray = {};
		lines[i].ParseIntoArray(stringArray, TEXT(","), false);

		// Put elements into lookup table
		FString key = stringArray[0];
		if (stringArray.Num() == 3) {
			// density, speed of sound
			float z = FCString::Atof(*stringArray[1]) * FCString::Atof(*stringArray[2]);
			materials.Add(key, z);
		}
	}
}

Octree* Octree::makeEnvOctreeRoot() {
	// Get caching/loading location
	FString filePath = FPaths::ProjectDir() + "Octrees/" + World->GetMapName();
	filePath +=
		"/min" + FString::FromInt(OctreeMin) + "_max" + FString::FromInt(OctreeMax);
	FString rootFile = filePath + "/" + "roots.json";

	// load
	Octree* root = new Octree(EnvCenter, OctreeRoot, rootFile);
	root->makeTill = Octree::OctreeMax;
	root->load();

	UE_LOG(LogHolodeck, Log, TEXT("Octree: Before line 117"));
	// set filename/makeTill for all OctreeMax nodes
	std::function<void(Octree*)> fix;
	fix = [&filePath, &fix](Octree* tree) {
		if (tree->size == Octree::OctreeMax) {
			tree->makeTill = Octree::OctreeMin;
			tree->file = filePath + "/" + FString::FromInt((int)tree->loc.X) + "_"
				+ FString::FromInt((int)tree->loc.Y) + "_"
				+ FString::FromInt((int)tree->loc.Z) + ".json";
		} else {
			for (Octree* l : tree->leaves) {
				fix(l);
			}
		}
	};
	fix(root);

	UE_LOG(LogHolodeck, Log, TEXT("Octree::Made Octree root"));

	return root;
}

Octree* Octree::makeOctree(
	FVector center,
	float	octreeSize,
	float	octreeMin,
	FString actorName) {
	FHitResult hit = FHitResult();
	bool	   occup;
	if (octreeSize == Octree::OctreeMin || actorName != "") {
		occup = World->SweepSingleByChannel(
			hit,
			center,
			center + FVector(0.01, 0.01, 0.01),
			FQuat::Identity,
			ECollisionChannel::ECC_WorldStatic,
			FCollisionShape::MakeBox(FVector(octreeSize / 2)),
			params);
	} else {
		occup = World->OverlapBlockingTestByChannel(
			center,
			FQuat::Identity,
			ECollisionChannel::ECC_WorldStatic,
			FCollisionShape::MakeBox(FVector(octreeSize / 2)),
			params);
	}

	// if we're making for an actor, make sure we're hitting it and not something
	// else
	if (occup && actorName != "" && actorName != hit.GetActor()->GetName()) {
		occup = false;
	}

	// if it's occupied
	if (occup) {
		// Check if it's full
		bool  full = true;
		// check to see if each corner is overlapping
		float distToCorner = octreeSize / 2 - cornerSize;
		for (FVector side : sides) {
			if (!full)
				break;
			full = World->OverlapBlockingTestByChannel(
				center + (side * distToCorner),
				FQuat::Identity,
				ECollisionChannel::ECC_WorldStatic,
				FCollisionShape::MakeBox(FVector(cornerSize)),
				params);
		}
		for (FVector corner : corners) {
			if (!full)
				break;
			full = World->OverlapBlockingTestByChannel(
				center + (corner * distToCorner),
				FQuat::Identity,
				ECollisionChannel::ECC_WorldStatic,
				FCollisionShape::MakeBox(FVector(cornerSize)),
				params);
		}

		if (!full) {
			// make a tree to insert
			Octree* child = new Octree(center, octreeSize);

			// if it still needs to be broken down, iterate through corners
			if (octreeSize > octreeMin) {
				for (FVector off : corners) {
					Octree* l = makeOctree(
						center + (off * octreeSize / 4),
						octreeSize / 2,
						octreeMin,
						actorName);
					if (l)
						child->leaves.Add(l);
				}
			}

			// if it's all the way broken down, save the normal
			else if (octreeSize == Octree::OctreeMin) {
				child->normal = hit.Normal;

				// Get material (there is tons of these!)
				child->fillMaterialProperties(getMaterialName(hit));

				// clean normal
				if (isnan(child->normal.X))
					child->normal.X = sign(child->normal.X);
				if (isnan(child->normal.Y))
					child->normal.Y = sign(child->normal.Y);
				if (isnan(child->normal.Z))
					child->normal.Z = sign(child->normal.Z);
				if (hit.Normal.ContainsNaN()) {
					UE_LOG(
						LogHolodeck,
						Warning,
						TEXT("Octree: Found position: %s"),
						*child->loc.ToString());
					UE_LOG(
						LogHolodeck,
						Warning,
						TEXT("Octree: Found nan: %s"),
						*child->normal.ToString());
				}
				// DrawDebugLine(World, center, center+hit.Normal*OctreeMin/2,
				// FColor::Blue, true, 100, ECC_WorldStatic, 1.f); DrawDebugBox(World,
				// center, FVector(octreeSize/2), FColor::Green, true, 2,
				// ECC_WorldStatic, 5.0f);
			}

			return child;
		}
	}

	// DrawDebugBox(World, center, FVector(octreeSize/2), FColor::Red, true, 2,
	// ECC_WorldStatic, 5.0f);
	return nullptr;
}

int Octree::numLeaves() {
	if (leaves.Num() == 0) {
		return 1;
	} else {
		int num = 1;
		for (Octree* leaf : leaves) {
			num += leaf->numLeaves();
		}
		return num;
	}
}

void Octree::toJson() {
	// make directory
	FFileManagerGeneric().MakeDirectory(*FPaths::GetPath(file), true);

	FJsonDomBuilder::FObject doc;

	// fill in buffer
	toJson(doc);

	std::ofstream fp(TCHAR_TO_UTF8(*file));

	fp << TCHAR_TO_ANSI(*(doc.ToString()));
	fp.close();
}

void Octree::toJson(FJsonDomBuilder::FObject& doc) {

	FJsonDomBuilder::FArray pArray;
	pArray.Add(loc[0], loc[1], loc[2]);

	doc.Set("p", pArray);

	if (leaves.Num() != 0) {

		FJsonDomBuilder::FArray leafArray;

		for (Octree* l : leaves) {
			leafArray.Add(l->createLeafObject());
		}

		doc.Set("l", leafArray);
	}

	if (size == OctreeMin) {
		FJsonDomBuilder::FArray normalArray;

		normalArray.Add(normal[0], normal[1], normal[2]);
		doc.Set("n", normalArray);
		doc.Set("m", TCHAR_TO_ANSI(*material));
	}
}

FJsonDomBuilder::FObject Octree::createLeafObject() {
	FJsonDomBuilder::FObject leafObject;
	FJsonDomBuilder::FArray	 pArray;
	pArray.Add(loc[0], loc[1], loc[2]);
	leafObject.Set("p", pArray);

	if (leaves.Num() != 0) {
		FJsonDomBuilder::FArray leafArray;

		for (Octree* l : leaves) {
			leafArray.Add(l->createLeafObject());
		}

		leafObject.Set("l", leafArray);
	}

	if (size == OctreeMin) {
		FJsonDomBuilder::FArray normalArray;

		normalArray.Add(normal[0], normal[1], normal[2]);
		leafObject.Set("n", normalArray);
		leafObject.Set("m", TCHAR_TO_ANSI(*material));
	}

	return leafObject;
}

void Octree::load() {
	// if it's not already loaded
	if (leaves.Num() == 0) {
		// if it's been saved as a json, load it
		if (FPaths::FileExists(file)) {
			UE_LOG(LogHolodeck, Log, TEXT("Octree: Loading Octree %s"), *file);
			// load file to a string

			gason::JsonAllocator allocator;
			std::ifstream		 t(TCHAR_TO_ANSI(*file));
			std::string			 str(
				 (std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
			char* source = &str[0];

			// process json
			char*			 endptr;
			gason::JsonValue json;
			int status = gason::jsonParse(source, &endptr, &json, allocator);

			// load in leaves
			for (gason::JsonNode* o : json) {
				if (o->key[0] == 'l') {
					for (gason::JsonNode* l : o->value) {
						loadJson(l->value, leaves, size / 2);
					}
				}
			}
		}

		// Otherwise build it & save for later
		else {
			UE_LOG(LogHolodeck, Log, TEXT("Octree: Making Octree %s"), *file);
			for (FVector off : corners) {
				Octree* l = makeOctree(loc + (off * size / 4), size / 2, makeTill);
				if (l)
					leaves.Add(l);
			}
			toJson();
		}
	}
}

void Octree::loadJson(gason::JsonValue& json, TArray<Octree*>& parent, float size) {
	Octree* child = new Octree;
	for (gason::JsonNode* o : json) {
		if (o->key[0] == 'p') {
			gason::JsonNode* arr = o->value.toNode();
			child->loc = FVector(
				arr->value.toNumber(),
				arr->next->value.toNumber(),
				arr->next->next->value.toNumber());
		}
		if (o->key[0] == 'l') {
			for (gason::JsonNode* l : o->value) {
				loadJson(l->value, child->leaves, size / 2);
			}
		}
		if (o->key[0] == 'n') {
			gason::JsonNode* arr = o->value.toNode();
			child->normal = FVector(
				arr->value.toNumber(),
				arr->next->value.toNumber(),
				arr->next->next->value.toNumber());
		}
		if (o->key[0] == 'm') {
			child->fillMaterialProperties(FString(o->value.toString()));
		}
	}
	child->size = size;
	parent.Add(child);
}

void Octree::unload() {
	if (!isAgent && leaves.Num() != 0) {
		// if we need to unload children
		if (size > Octree::OctreeMax) {
			for (Octree* leaf : leaves)
				leaf->unload();
		}

		// if we need to unload this one
		else if (size == Octree::OctreeMax) {
			UE_LOG(LogHolodeck, Log, TEXT("Octree: Unloading Octree %s"), *file);
			for (Octree* leaf : leaves)
				delete leaf;
			leaves.Reset();
		}
	}
}

void Octree::fillMaterialProperties(FString mat) {
	material = mat;
	float matProp;
	bool  found = materials.Find(material, matProp);
	if (!found) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"Octree: Missing material information for %s, adding in blank "
				"row to csv"),
			*this->material);

		// Add default line to material file to fill in later
		FString filePath = FPaths::ProjectDir() + "../../materials.csv";
		FString line = "\n" + material + ", 10000, 10000";
		FFileHelper::SaveStringToFile(
			line,
			*filePath,
			FFileHelper::EEncodingOptions::AutoDetect,
			&IFileManager::Get(),
			EFileWrite::FILEWRITE_Append);

		// Default to something really high to get full reflection for this time
		z = 10000 * 10000;
		materials.Add(material, z);
	} else {
		z = matProp;
	}
}

FString Octree::getMaterialName(FHitResult hit) {
	// Get staticmesh material
	// UMaterialInterface* mat =
	// hit.GetComponent()->GetMaterial(hit.ElementIndex); if(mat != nullptr){
	// 	return mat->GetFName().ToString();
	// }

	// // If not staticmesh, get landscape material
	// AActor* actor = hit.GetActor();
	// ALandscapeProxy* landscape = reinterpret_cast<ALandscapeProxy*>(actor);
	// mat = landscape->LandscapeMaterial;

	// if(mat != nullptr){
	// 	return mat->GetFName().ToString();
	// }

	// // If we have extra issues getting material
	// UE_LOG(LogHolodeck, Warning, TEXT("Octree: Couldn't get material name for
	// an octree leaf, putting as MaterialNotFound"));
	return "MaterialNotFound";
}