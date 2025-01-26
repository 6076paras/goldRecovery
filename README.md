## **Gold Purification Process**

Gold purification is a multi-stage process used to extract pure gold from ore, focusing on flotation and leaching stages.

### 1. **Flotation Process (First Stage)**

Flotation separates gold from gangue based on surface properties. Gold becomes hydrophobic, attaching to air bubbles and floating to the surface, while gangue stays submerged.

#### Flotation Steps:
- Ore is crushed and mixed with water to form a slurry.
- Flotation chemicals are added to separate gold from gangue.
- Air bubbles cause gold to float, forming a froth, which is skimmed off as **rougher concentrate**.
- Remaining material, **rougher tails**, has a lower gold concentration and is either discarded or further treated.

### 2. **First Stage of Leaching (Cyanidation)**

Cyanidation uses cyanide to dissolve gold from the concentrate into a gold-cyanide complex.

#### Cyanidation Steps:
- Flotation concentrate is mixed with sodium cyanide.
- Cyanide leaches gold into a liquid form, separating it from gangue.

### 3. **Second Stage of Leaching (Activated Carbon or Zinc Precipitation)**

Gold is recovered from the cyanide solution using activated carbon or zinc.

#### Activated Carbon Adsorption:
- Cyanide solution is passed through activated carbon, adsorbing gold.
- Gold is stripped from the carbon and refined.

#### Zinc Precipitation:
- Zinc is added to the cyanide solution, causing gold to precipitate.
- Gold is filtered out and refined.



#### Data Description
The column names in the dataset follow the structure:

```
[stage].[parameter_type].[parameter_name]
```
Where:

- **[stage]** refers to the specific stage in the process:
  - `rougher` — flotation stage
  - `primary_cleaner` — primary purification stage
  - `secondary_cleaner` — secondary purification stage
  - `final` — final product characteristics

- **[parameter_type]** refers to the type of the parameter:
  - `input` — raw material parameters
  - `output` — product parameters
  - `state` — parameters that characterize the current state of the stage
  - `calculation` — derived or calculation-based parameters

- **[parameter_name]** refers to the specific parameter being measured. For full description refer to [`parameter_names.md`]((https://github.com/paras-p/gold-recovery/blob/main/goldrecovery/data/parameter_names.md) "goldrecovery/data/parameter_names.md") 

The following image visually represents the gold recovery process workflow, highlighting stages like flotation, primary purification, secondary purification, and final product characteristics. It also uses example variables to illustrate how data corresponds to stages and parameter types, such as gold content for input and output

<div align="center">
<img src="https://github.com/6076paras/goldRecovery/blob/3b1c9ce41e9eae4052cd8e2777a6a65f8c34a23a/assets/technological_process.png" width="50%" height="50%" alt="Process Description" />

</div>