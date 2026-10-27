# Voici les commandes utilisées pour obtenir le plan d'obs avec le même setup que Thomas Culino 

# 2 visit:
```bash
m4opt schedule skymaps/glg_healpix_all_bn261013582.fit /home/liteul/phd/events/GCN261013/sch_tom_2v.ecsv \
--mission ztf --deadline 76min  --time-step 1.0min --event-time 2026-10-14T10:41:00 --timelimit 1000s \
--bandpass g --bandpass r --visits 2 --nside 512 --max-fields 50 --jobs 0 \
--write-progress /home/liteul/phd/events/GCN261013/PROGRESS.ecsv --exptime-min 300s --exptime-min 300s \
--write-model /home/liteul/phd/events/GCN261013/MODEL_thom_2v.lp
```

## Animate:
```bash
m4opt animate /home/liteul/phd/events/GCN261013/sch_tom_2v.ecsv \
--time-step 10min --still /home/liteul/phd/events/GCN261013/still_t_2v.pdf --dpi 180 \
/home/liteul/phd/events/GCN261013/movie_t_2v.gif

```

# 1 visit:
```bash
m4opt schedule skymaps/glg_healpix_all_bn261013582.fit /home/liteul/phd/events/GCN261013/sch_tom_1v.ecsv \
--mission ztf --deadline 76min  --time-step 1.0min --event-time 2026-10-14T10:41:00 --timelimit 1000s \
--bandpass g --bandpass r --visits 1 --nside 512 --max-fields 50 --jobs 0 \
--write-progress /home/liteul/phd/events/GCN261013/PROGRESS.ecsv --exptime-min 300s --exptime-min 300s \
--write-model /home/liteul/phd/events/GCN261013/MODEL_thom_1v.lp
```

## Animate:
```bash
m4opt animate /home/liteul/phd/events/GCN261013/sch_tom_1v.ecsv \
--time-step 10min --still /home/liteul/phd/events/GCN261013/still_t_1v.pdf --dpi 180 \
/home/liteul/phd/events/GCN261013/movie_t_1v.gif

```