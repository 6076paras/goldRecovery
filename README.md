#### Gold Purification Recovery Optimization and Process Control


This project focuses on optimizing gold recovery efficiency in the purification process using machine learning techniques. By analyzing data from various stages of the flotation process, the developed models predict and improve recovery rates while optimizing reagent consumption. The project leverages Zyfra's extensive dataset to create a digital twin of the purification process, enabling risk-free parameter experimentation and process optimization.

For a detailed analysis and technical report of this project, check out the blog post:
[Machine Learning Integration for Gold Recovery Process Control](https://6076paras.github.io/blog/2025/gold-recovery-optimization/)


<!-- ### **Gold Purification Process**

Gold purification is a multi-stage process used to extract pure gold from ore, focusing on flotation and leaching stages.

#### 1. Flotation Process

Flotation separates gold from gangue based on surface properties. Gold becomes hydrophobic, attaching to air bubbles and floating to the surface, while gangue stays submerged.

#### Flotation Steps:
- Ore is crushed and mixed with water to form a slurry.
- Flotation chemicals are added to separate gold from gangue.
- Air bubbles cause gold to float, forming a froth, which is skimmed off as **rougher concentrate**.
- Remaining material, **rougher tails**, has a lower gold concentration and is either discarded or further treated.

#### 2. First Stage of Leaching

Cyanidation uses cyanide to dissolve gold from the concentrate into a gold-cyanide complex.

#### Cyanidation Steps:
- Flotation concentrate is mixed with sodium cyanide.
- Cyanide leaches gold into a liquid form, separating it from gangue.

#### 3. Second Stage of Leaching

Gold is recovered from the cyanide solution using activated carbon or zinc.

#### Activated Carbon Adsorption:
- Cyanide solution is passed through activated carbon, adsorbing gold.
- Gold is stripped from the carbon and refined.

#### Zinc Precipitation:
- Zinc is added to the cyanide solution, causing gold to precipitate.
- Gold is filtered out and refined.



### **Data Description**
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

- **[parameter_name]** refers to the specific parameter being measured. For a full description, refer to the [`parameter_names.md`](command:_github.copilot.openRelativePath?%5B%22goldrecovery%2Fdata%2Fparameter_names.md%22%5D "goldrecovery/data/parameter_names.md") file.

The following image visually represents the gold recovery process workflow, highlighting stages like flotation, primary purification, secondary purification, and final product characteristics. It also uses example variables to illustrate how data corresponds to stages and parameter types, such as gold content.

<div align="center">
<img src="https://github.com/6076paras/goldRecovery/blob/7cfb134d6b13caffe8275ae1d06b5e103509b1b7/assets/data_description.png" width="50%" height="50%" alt="Process Description" />

</div> -->