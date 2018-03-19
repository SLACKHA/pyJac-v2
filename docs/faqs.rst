Frequently asked Questions
##########################

I get an error stating:
```
OSError: dlopen: cannot load any more object with static TLS
```
when running validation testing.  This has been known to occur in SLURM or related
job-batch systems when loading the Adept library for auto-differentiation execution.
To work around this, [you should set the environment variable `LD_PRELOAD`](https://stackoverflow.com/a/45640803/1667311), e.g.:
```
LD_PRELOAD=/path/to/adept/install/lib/libadept.so python -m pyjac.functional_tester ...
```
