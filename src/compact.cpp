#include "internal.hpp"

namespace CaDiCaL {

/*------------------------------------------------------------------------*/

// Compactifying removes holes generated by inactive variables (fixed,
// eliminated or substituted) by mapping active variables indices down to a
// contiguous interval of indices.

/*------------------------------------------------------------------------*/

bool Internal::compactifying () {
  if (level) return false;
  if (!opts.simplify) return false;
  if (!stats.compacts) return true; // TODO remove
  if (!opts.compact) return false;
  if (stats.conflicts < lim.compact) return false;
  int inactive = max_var - active_variables ();
  assert (inactive >= 0);
  if (inactive < opts.compactmin) return false;
  return inactive >= opts.compactlim * max_var;
}

/*------------------------------------------------------------------------*/

// Map old internal literal 'SRC' to new internal literal 'DST'.  This would
// be trivially just a look-up into the 'map' created in 'compact' (caring
// about signedness of 'SRC' though), except that fixed variables have all
// to be mapped to the first fixed variable 'first_fixed', which makes it
// more tricky.
//
#define MAP_LIT(SRC,DST) \
do { \
  int OLD = (SRC); \
  assert (OLD), assert (abs (OLD) <= max_var); \
  int RES = map[abs (OLD)]; \
  if (!RES) { \
    assert (!level); \
    const int TMP = val (OLD); \
    if (TMP) { \
      assert (first_fixed); \
      RES = map_first_fixed; \
      if (TMP != first_fixed_val) RES = -RES; \
    } \
  } else if ((OLD) < 0) RES = -RES; \
  assert (abs (RES) <= new_max_var); \
  (DST) = RES; \
} while (0)

#define MAP_ARRAY(TYPE,NAME) \
do { \
  for (int SRC = 1; SRC <= max_var; SRC++) { \
    const int DST = map[SRC]; \
    if (!DST) continue; \
    assert (0 < DST), assert (DST <= SRC); \
    assert (DST > 0); \
    NAME[DST] = NAME[SRC]; \
  } \
  SHRINK (NAME, TYPE, vsize, new_vsize); \
  PRINT ("mapped '" # NAME "'"); \
} while (0)

// Same as 'MAP_ARRAY' but two sided (positive & negative literal).
//
#define MAP2_ARRAY(TYPE,NAME) \
do { \
  for (int SRC = 1; SRC <= max_var; SRC++) { \
    const int DST = map[SRC]; \
    if (!DST) continue; \
    assert (0 < DST), assert (DST <= SRC); \
    NAME[2*DST] = NAME[2*SRC]; \
    NAME[2*DST+1] = NAME[2*SRC+1]; \
  } \
  SHRINK (NAME, TYPE, 2*vsize, 2*new_vsize); \
  PRINT ("mapped '" # NAME "'"); \
} while (0)

// Map a 'vector<int>' of literals, flush inactive literals, resize and
// shrink it to fit its new size after flushing.
//
#define MAP_AND_FLUSH_INT_VECTOR(V) \
do { \
  const const_int_iterator end = V.end (); \
  int_iterator j = V.begin (); \
  const_int_iterator i; \
  for (i = j; i != end; i++) { \
    const int SRC = *i; \
    int DST = map[abs (SRC)]; \
    assert (abs (DST) <= abs (SRC)); \
    if (!DST) continue; \
    if (SRC < 0) DST = -DST; \
    *j++ = DST; \
  } \
  V.resize (j - V.begin ()); \
  shrink_vector (V); \
  PRINT ("mapped '" # V "'"); \
} while (0)

/*------------------------------------------------------------------------*/

#if 1
#define PRINT(MSG) \
do { \
  if (!opts.verbose) break; \
  printf ("c %s %.0f MB\n", (MSG), current_resident_set_size ()/(double)(1<<20) ); \
  fflush (stdout); \
} while (0)
#else
#define PRINT(MSG) do { } while (0)
#endif

void Internal::compact () {

  PRINT ("BEFORE");

  START (compact);
  stats.compacts++;

  assert (!level);
  assert (!unsat);
  assert (!conflict);
  assert (clause.empty ());
  assert (levels.empty ());
  assert (analyzed.empty ());
  assert (minimized.empty ());
  assert (control.size () == 1);
  assert (resolved.empty ());
  assert (propagated == trail.size ());
  assert (active_variables () < max_var);

  if (lim.fixed_at_last_collect < stats.all.fixed) {
    LOG ("forcing garbage collection");
    garbage_collection ();
  }

  // Remember whether this was 'triggered' from 'compactifying', since only
  // then we should increase the conflict limit.
  //
  const bool triggered = compactifying ();

  // We produce a compactifying garbage collector like map of old 'src' to
  // new 'dst' variables.  Inactive variables are just skipped except for
  // fixed ones which will be mapped to the first fixed variable (in the
  // appropriate phase).  This avoids to handle the case 'fixed value'
  // seperately as it is done in Lingeling, where fixed variables are
  // mapped to the internal variable '1'.
  //
  int * map, new_max_var = 0, first_fixed = 0, map_first_fixed = 0;
  NEW (map, int, max_var + 1);
  map[0] = 0;
  for (int src = 1; src <= max_var; src++) {
    const Flags & f = flags (src);
    if (f.active ()) map[src] = ++new_max_var;
    else if (!f.fixed () || first_fixed) map[src] = 0;
    else map[first_fixed = src] = map_first_fixed = ++new_max_var;
  }

  const int first_fixed_val = first_fixed ? val (first_fixed) : 0;

  if (first_fixed)
    LOG ("found first fixed %d", sign (first_fixed_val)*first_fixed);
  else LOG ("no variable fixed");

  const size_t new_vsize = new_max_var + 1;  // Adjust to fit 'new_max_var'.

  PRINT ("generated 'map'");

  /*----------------------------------------------------------------------*/
  // In this first part we only map stuff without reallocation.
  /*----------------------------------------------------------------------*/

  // Flush the external indices.  This has to occur before we map 'vals'.
  {
    for (int eidx = 1; eidx <= external->max_var; eidx++) {
      int src = external->e2i[eidx], dst;
      if (!src) continue;
      MAP_LIT (src, dst);
      LOG ("compact %ld maps external %d to internal %d from %d",
        stats.compacts, eidx, dst, src);
      external->e2i[eidx] = dst;
    }
  }

  PRINT ("mapped 'i2e'");

  // Map the literals in all clauses.
  {
    const const_clause_iterator end = clauses.end ();
    const_clause_iterator i;
    for (i = clauses.begin (); i != end; i++) {
      Clause * c = *i;
      const const_literal_iterator eoc = c->end ();
      literal_iterator j;
      for (j = c->begin (); j != eoc; j++) {
	const int src = *j;
	assert (!val (src));
	int dst;
	MAP_LIT (src, dst);
	assert (dst || c->garbage);
	*j = dst;
      }
    }
  }

  PRINT ("mapped 'clauses'");

  // Map the blocking literals in all watches.
  //
  if (watches ()) {
    for (int idx = 1; idx <= max_var; idx++) {
      for (int sign = -1; sign <= 1; sign += 2) {
	const int lit = sign*idx;
	Watches & ws = watches (lit);
	const const_watch_iterator end = ws.end ();
	watch_iterator i;
	for (i = ws.begin (); i != end; i++)
	  MAP_LIT (i->blit, i->blit);
      }
    }
  }

  PRINT ("mapped 'blits'");

  // We first flush inactive variables and map the links in the queue.  This
  // has to be done before we map the actual links data structure 'ltab'.
  {
    int prev = 0, mapped_prev = 0, next;
    for (int idx = queue.first; idx; idx = next) {
      Link * l = ltab + idx;
      next = l->next;
      if (idx == first_fixed) continue;
      const int dst = map[idx];
      if (!dst) continue;
      assert (active (idx));
      if (prev) ltab[prev].next = dst; else queue.first = dst;
      l->prev = mapped_prev;
      mapped_prev = dst;
      prev = idx;
    }
    if (prev) ltab[prev].next = 0; else queue.first = 0;
    queue.unassigned = queue.last = mapped_prev;
  }

  PRINT ("mapped 'queue'");

  /*----------------------------------------------------------------------*/
  // In second part we map and flush arrays.
  /*----------------------------------------------------------------------*/

  MAP_AND_FLUSH_INT_VECTOR (trail);
  propagated = trail.size ();
  if (first_fixed) {
    assert (trail.size () == 1);
    var (first_fixed).trail = 0;		// before mapping 'vtab'
  } else assert (trail.empty ());

  if (!probes.empty ()) MAP_AND_FLUSH_INT_VECTOR (probes);

  /*----------------------------------------------------------------------*/
  // In third part we not only map stuff but also reallocate memory.
  /*----------------------------------------------------------------------*/

  // Now we continue in reverse order of allocated bytes, e.g., see
  // 'Internal::enlarge' which reallocates in order of allocated bytes.

  MAP_ARRAY (Flags, ftab);
  MAP_ARRAY (signed_char, marks);
  MAP_ARRAY (signed_char, phases);

#if 1
  // Special case for 'val' as always since for 'val' we trade branch less
  // code for memory and always allocated an [-maxvar,...,maxvar] array.
  {
    signed char * new_vals = new signed char [2*new_vsize];
    new_vals += new_vsize;
    for (int src = -max_var; src <= -1; src++)
      new_vals[-map[-src]] = vals[src];
    for (int src = 1; src <= max_var; src++)
      new_vals[map[src]] = vals[src];
    new_vals[0] = 0;
    vals -= vsize;
    delete [] vals;
    vals = new_vals;
  }
#else
#endif

  PRINT ("mapped 'vals'");

  MAP_ARRAY (int, i2e);
  MAP2_ARRAY (int, ptab);
  MAP_ARRAY (long, btab);
  if (ntab2) MAP_ARRAY (long, ntab2);
  MAP_ARRAY (Link, ltab);
  MAP_ARRAY (Var, vtab);
  if (ntab) MAP2_ARRAY (long, ntab);
  if (wtab) MAP2_ARRAY (Watches, wtab);
  if (otab) MAP2_ARRAY (Occs, otab);
  if (big) MAP2_ARRAY (Bins, big);

  // The simplest way to map the elimination schedule is to get all elements
  // from the heap and reinsert them.  This could be slightly improved in
  // terms of speed if we add a 'flush (int * map)' function to 'Heap', but
  // is pretty complicated and would require that the 'Heap' knows that
  // mapped elements with 'zero' destination should be flushed.  Note that
  // we use stable heap sorting.
  //
  if (!esched.empty ()) {
    vector<int> saved;
    while (!esched.empty ()) {
      const int src = esched.front ();
      esched.pop_front ();
      const int dst = map [src];
      if (dst && src != first_fixed) saved.push_back (dst);
    }
    esched.clear ();
    const const_int_iterator end = saved.end ();
    const_int_iterator i;
    for (i = saved.begin (); i != end; i++)
      esched.push_back (*i);
    esched.shrink ();
  }

  PRINT ("mapped 'esched'");

  /*----------------------------------------------------------------------*/

  DELETE (map, int, max_var);

  VRB ("compact", stats.compacts,
    "reducing internal variables from %d to %d",
    max_var, new_max_var);

  max_var = new_max_var;
  vsize = new_vsize;

  stats.now.fixed = first_fixed ? 1 : 0;
  stats.now.substituted = stats.now.eliminated = 0;

  if (triggered) inc.compact += opts.compactint;
  lim.compact = stats.conflicts + inc.compact;
  report ('c');
  STOP (compact);

  PRINT ("AFTER");
}

};
