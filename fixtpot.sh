#!/bin/bash
fname=`basename $1`
pklfname=`echo $fname|sed 's/\..*/.pkl/'`
sed -i "s/'PATH\/TO\/DATA\/FILE'/sys.argv[1]/;s/COLUMN_SEPARATOR/,/;s/\['target'\]/['$2']/;s/'target'/columns='$2'/" $1
sed -i "1iimport sys" $1
sed -i "1iimport joblib" $1
sed -i '$d' $1
echo "joblib.dump(exported_pipeline,'$pklfname')" >>$1
