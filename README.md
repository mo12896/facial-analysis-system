<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
  </a>

  <h2 align="center">Prediciting Well-Being in Team Collaboration from Video Data Using Machine Learning</h2>

  <p align="center">
    The main repository for my master thesis.
    <br />
    <br />
  </p>
</div>

<img src="./docs/animation.gif" widt="920" height="500" alt="animated" />

<!-- TABLE OF CONTENTS
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->


<!-- ABOUT THE PROJECT -->
## About The Project

This repository is dedicated to my master thesis, which focuses on predicting individual well-being, as defined by the PERMA framework, in team collaboration. The objective of this research is to extract non-verbal cues such as facial emotions, gaze and head motion patterns from individual team members using novel AI tools for face detection, tracking, re-identification, facial emotion recognition and head pose estimation for example. The generated features will then be used to predict the PERMA scores of each team member.

PERMA is a metric that measures subjective well-being, and it is an essential part of this research. The overall goal is to increase the well-being of individuals and teams by better understanding the factors that contribute to well-being in a team setting.

Feel free to explore the repository and ask any questions or share ideas. Your feedback and collaboration are always welcome. I am looking forward to sharing my research with you and making a positive impact in the field of team dynamics.

If you have any questions or ideas, please don't hesitate to get in touch. I am always open to feedback and suggestions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ### Built With

* [Python][Python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow the following steps.

### Prerequisites

To harness the full power of CUDA GPU acceleration during inference, ensure that your machine is equipped with a NVIDIA Graphics Card (GPU).

### Installation


1. Clone the repo and change into the directory:
   ```sh
   git clone https://github.com/mo12896/facial-analysis-system.git
   cd facial-analysis-system
   ```
2. Go into working directory and create virtual environment using conda
    ```sh
    cd facial-analysis-system
    conda create -n facesys python=3.8
    conda activate facesys
    ```
    ... or create virtual environment using venv
    ```sh
    cd facial-analysis-system
    python3.8 -m venv facesys
    source facesys/bin/activate
    ```
3. Run the setup script to install all dependencies:
   ```sh
   chmod +x setup.sh
   bash setup.sh
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## How to Use it
Place the video file in the folder `data/input` and write the full name of the video file alongside the video format in the the config file `config/config.yaml`. The video file must be in the correct format, i.e. .mp4 or .avi and contain video recordings of team collaboration. To enable the 3D gaze pattern estimation framework, the video must be captured with a 360° camera in the "two 180° images" mode.


### Pre-Processing
Set the relevant parameters in the config file `config/config.yaml` and run the following command, which starts the pre-processing script to generate the template embedding database for the relevant team members. The pre-processing step is optional and can be skipped if the database already exists. Different approaches to generate the template embeddings are imaginable and can be exchanged with the proposed approach. The following script runs the template generation process outlined in the master thesis and stores the images and templates database under `data/<video_name>/utils/`:

```sh
python main.py --mode 0
```


### Analysis Pipeline

Set the relevant parameters in the config file `config/config.yaml` and run the facial analysis pipeline. The pipeline can be used in three main steps, which can be run in sequences of different length. Each downstream step depends on the previous step. The steps can each be run seperately - if the previous steps have already been run - or directly in sequences, i.e. one of these combinations: ["0", "1", "2", "3", "01", "012", "0123", "12", "23", "123"], where 0 is the preprocessing step. The single steps are defined as follows:

1. **Facial Analysis [1]**: Extracts facial features from a .mp4 or .avi video file. The extracted features are stored in the folder `data/<video_name>/analysis_results/`:
   - CSV file: The CSV-file contains the facial features of each frame in the video file.
   - MP4 file (optinally): The MP4 file can be generated optionally and contains the video with the extracted facial features for visual verficiation. Can be invoked by setting the `-o` flag in the command.
2. **Feature Extraction [2]**: Extracts features from the facial features stored in the database. The extracted features are stored in the folder `data/<video_name>/extraction_results/`.
    - CSV file: The CSV-file contains the final features of each team member, visible in the video file.
    - PNG files (optinally): The PNG files can be generated optionally and contain the extracted features for visual verficiation.
3. **PERMA Prediction [3]**: Predicts the PERMA score of each team member based on the extracted features. The final predictions are stored in the folder `data/<video_name>/prediction_results/`.
    - CSV file: The CSV-file contains the predicted PERMA score of each team member.
    - PNG files (optinally): The PNG files can be generated optionally and contain the predicted PERMA score for visual verficiation.

As an exmaple, running all three steps in sequence can be done by running the following command:

```sh
python main.py --mode 123
```



<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Moritz Müller - moritz1996.mueller@gmail.com

Project Link: [Facial Analysis System](https://github.com/mo12896/facial-analysis-system)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/mo12896/emotion-recognition/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/mo12896/emotion-recognition/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/mo12896/emotion-recognition/LICENSE.txt


[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[Python-url]: https://www.python.org/
[ONNXGPU-url]: https://pypi.org/project/onnxruntime-gpu/
