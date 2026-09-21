DTOOLS = dtools

ENV = test

include .env
ifdef INSTANCES
	INSTANCES := -instances $(INSTANCES)
endif

clean:
	rm -rf release

dtest:
	rm -rf release
	mkdir -p release/bin
	mkdir -p release/conf
	mkdir -p release/script
	find ./ | grep -v .git | grep -v .venv | grep -v .idea | grep -v release | grep -v ret | grep -v .env | cpio -pdm release/bin/
	cd release && tar czf release.tgz bin conf script && $(DTOOLS) dpatch -env $(ENV) -app $(APP) -server $(SERVER) -tgz release.tgz  -user $(USER) $(INSTANCES) -increment true

t:
	make clean
	make dtest
	make clean
