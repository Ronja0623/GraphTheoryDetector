all: models env

models:
	unzip ./FC_Model/other_FC_model.zip -d ./FC_Model/
	unzip ./MD_Model/other_MD_model.zip -d ./MD_Model/

env:
	./env.sh