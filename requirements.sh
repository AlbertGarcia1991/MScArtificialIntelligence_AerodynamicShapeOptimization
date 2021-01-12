 #!/bin/bash 

sudo apt-get update -y

# install GMSH
sudo apt-get install -y gmsh

# install OpenFOAM
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get -y install openfoam7

sudo apt-get upgrade

