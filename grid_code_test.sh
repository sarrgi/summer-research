#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
# I know I have a directory here so I'll use it as my initial working directory
#
#$ -wd /vol/grid-solar/sgeusers/sargisfinl
#
# End of the setup directives
#
# Now let's do something useful, but first change into the job-specific directory that should
#  have been created for us
#
# Check we have somewhere to work now and if we don't, exit nicely.
#
if [ -d /local/tmp/sargisfinl/$JOB_ID.$SGE_TASK_ID ]; then
        cd /local/tmp/sargisfinl/$JOB_ID.$SGE_TASK_ID
else
       echo "Uh oh ! There's no job directory to change into "
       echo "Something is broken. I should inform the programmers"
       echo "Save some information that may be of use to them"
       echo "Here's LOCAL TMP "
       ls -la /local/tmp
       echo "AND LOCAL TMP sargisfinl "
       ls -la /local/tmp/sargisfinl
       echo "Exiting"
       exit 1
fi
#
# Now we are in the job-specific directory so now can do something useful
#

#
# Mail me at the b(eginning) and e(nd) of the job
#
# $ -M sargisfinl@myvuw.ac.nz
# $ -m be
#

# Stdout from programs and shell echos will go into the file
#    scriptname.o$JOB_ID
#  so we'll put a few things in there to help us see what went on
#
# echo ==UNAME==
# uname -n
# echo ==WHO AM I and GROUPS==
# id
# groups
# echo ==SGE_O_WORKDIR==
# echo $SGE_O_WORKDIR
# echo ==/LOCAL/TMP==
# ls -ltr /local/tmp/
# echo ==/VOL/GRID-SOLAR==
# ls -l /vol/grid-solar/sgeusers/
#
# OK, where are we starting from and what's the environment we're in
#
echo ==RUN HOME==
pwd
ls
echo ==ENV==
env
echo ==SET==
set
#
# echo == WHATS IN LOCAL/TMP ON THE MACHINE WE ARE RUNNING ON ==
# ls -ltra /local/tmp | tail
#
echo == WHATS IN LOCAL TMP sargisfinl JOB_ID AT THE START==
ls -la
#
# Copy the input file to the local directory
#
cp /vol/grid-solar/sgeusers/sargisfinl/mygp.py .
cp /vol/grid-solar/sgeusers/sargisfinl/slide.py .
cp /vol/grid-solar/sgeusers/sargisfinl/slideGP.py .
cp /vol/grid-solar/sgeusers/sargisfinl/test.py .
cp /vol/grid-solar/sgeusers/sargisfinl/output.txt .
echo ==WHATS THERE HAVING COPIED STUFF OVER AS INPUT==
ls -la
#
# run program
#
python slideGP.py $SGE_TASK_ID
#
echo ==AND NOW, HAVING DONE SOMTHING USEFUL AND CREATED SOME OUTPUT==
ls -la
#
# Now we move the output to a place to pick it up from later
#  (really should check that directory exists too, but this is just a test)
#
mkdir -p /vol/grid-solar/sgeusers/sargisfinl/$JOB_ID/$SGE_TASK_ID
cp output.txt  /vol/grid-solar/sgeusers/sargisfinl/$JOB_ID/$SGE_TASK_ID
#
echo "Ran through OK"
